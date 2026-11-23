# WritableTree.extend 故障恢复（整批原子边界）

本目录是本地故障注入复现与验证套件，与实现中的恢复判断完全独立：注入器和检查程序只使用
uproot 的公开接口（`recreate` / `mktree` / `extend` / `open`），不读取内部恢复标志，
也不以“是否捕获到异常”作为通过标准——所有结论都来自独立读句柄对落盘字节的核对。

## 文件

- `repro.py`：最小复现。未修复版本下，一次写入失败后重新打开文件会得到
  `DeserializationError`（树元数据与 basket 记录互相矛盾）；修复后失败批次被整体撤回，
  前缀与后续追加均正确。
- `injector.py`：故障注入器（普通文件 `FaultyFile` 与内存 file-like `FaultyBytesIO`）
  与独立核对函数 `check_prefix`。故障位置按“arm 之后的第 N 次 write/flush 调用”编号。
- `sweep.py`：穷举扫描器。对一次普通追加和一次触发元数据扩容的追加，遍历其全部
  write/flush 位置，各注入一次故障；file-like 后端取代表性位置。
- 仓库内正式回归：`tests/test_1524_writable_tree_batch_recovery.py`。

## 故障模型

每个用例只注入一次故障，之后读、写、flush 全部恢复正常。覆盖：

| 模式 | 行为 |
| --- | --- |
| `raise_before` | write 在写入任何字节前抛 `OSError` |
| `partial` | 先覆写请求的一部分字节，再抛 `OSError` |
| `short` | 返回小于请求长度的正数 |
| `zero` | 返回 0（无进展） |
| `flush` | flush 抛 `OSError` |

## 运行方法（干净环境复跑）

```bash
# 1. 准备 Python 3.12 环境（任选其一，以下用 uv）
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "numpy==2.2.6" "awkward==2.8.12" pytest
uv pip install --python .venv/bin/python -e .

# 2. 最小复现（修复后应全部通过）
.venv/bin/python -I dev/fault_recovery/repro.py

# 3. 穷举故障位置扫描（预期 0 failures）
.venv/bin/python -I dev/fault_recovery/sweep.py

# 4. 正式回归
.venv/bin/python -m pytest tests/test_1524_writable_tree_batch_recovery.py -q
```

## 选择的方案与备选权衡

需求约束：不复制整个旧 ROOT 文件、不读回全部历史事件、不在内存保留历史 basket；
每批额外驻留数据只能与“当前批 + 涉及的元数据”同阶，且随文件历史增长保持有界。

### 采用：延后发布 + 修改前留存（snapshot + 字节范围备份）

两层配合：

1. **物理层（`uproot.sink.file.FileSink`）**
   - 新增 `begin_transaction` / `commit_transaction` / `rollback_transaction`。
   - 事务期间每次写入前，先把目标字节区间的原始内容读出备份（区间按不相交、有序
     合并维护）；这只覆盖本批实际触及的位置——新 basket 落在此前空闲/文件尾区域，
     备份量为零，被覆写的只是树元数据、TFree、文件头、目录块等有界元数据。
   - `write` 改为“必须写满”：短写在重新 seek 到正确续写点后重试，零进展或非法
     返回值直接抛 `OSError`，不会静默生成残缺 basket。
   - 提交时先 flush 成功才丢弃备份；回滚时按备份把被覆写字节原样写回。
   - 对不可读回流的只写流（如 `fsspec` 的 `wb` 句柄）探测后自动降级：只保证
     全长度写检查与内存态回滚，不做字节恢复（与该类句柄的历史能力一致）。

2. **逻辑层（`uproot.writing._cascadetree.Tree.extend`）**
   - 输入规整与所有校验移到任何 I/O 之前；空批直接 no-op。校验失败和空批都不推进
     `num_entries` / `num_baskets`，也不分配空间。
   - I/O 前对树、各分支（fBasket* 数组、字节计数、写入游标）、FreeSegments
     （key/data/fileheader 全部内部字段）、目录（datakey、DirectoryData、
     DirectoryHeader）以及 `WritableFile._trees` 做深快照。
   - basket 写入、容量扩容（TTree 元数据搬迁 + 旧位置释放）、元数据更新全部在事务内
     完成；任一步抛错就先回滚字节再恢复内存态，使本批像从未发生过。
   - 成功返回即提交点：只有此刻边界才前移。

**代价**

- 已覆写字节：可恢复，因为覆写前留存了原字节；代价是事务期间多占一份“被触及元数据
  区间”的字节（与历史文件大小无关，只与本批触碰的元数据成比例）。
- 元数据搬迁：新树块、新目录块、新 TFree 的位置在失败后整体作废，旧块字节由备份恢复，
  旧位置在内存态恢复后重新归该对象所有；失败分配可能留下未引用空洞（需求允许），
  不会被后续分配误判为已提交数据或覆盖活对象。
- 空间归属：回滚后 FreeSegments 的 slices/end/key 位置全部回到快照值，与磁盘上恢复出的
  TFree 记录一致。
- flush 时机：提交必须等最后一次 flush 成功；flush 失败按批次失败处理（flush 不改字节，
  回滚只需恢复此前覆写的区间）。
- 每批额外开销：一次事务区间备份读 + 失败时一次恢复写；成功路径无额外写放大。

### 备选 1：整文件复制（写前 copy-on-side）

实现最简单、对任意部分覆写都天然可恢复，但每次追加都要复制整个 ROOT 文件，历史越大
每批成本越高，直接违反“不复制整个旧文件、准备量不随历史线性增长”的约束，被否决。

### 备选 2：只延后发布、不做字节留存（影子分配，永不就地覆写）

所有本批数据（含更新后的树元数据、目录、TFree、文件头）都写到新位置，成功后才一次性
切换引用；失败只留下垃圾字节。问题是文件头（固定 0 号位置）和目录头（稳定锚点）按
ROOT 格式必须就地更新，无法只靠“新位置 + 切引用”完成；要补齐就得为这些固定锚点再引入
修改前留存，最终退化为当前方案，且复杂度更高，故不采用。

### 备选 3：WAL（预写日志）

把所有变更先顺序写日志再 apply，恢复时重放/撤销。通用性最强，但需要新增日志记录格式、
重放状态机和崩溃后的打开路径，远超本问题范围，且与现有 ROOT 就地更新格式不兼容，
本次不引入新的对外事务 API，故不采用。

## 恢复范围与不保证的事项

保证：单次可恢复 I/O 异常（write 抛错、部分覆写、短写、零进展、flush 抛错，且故障只
发生一次）下，`extend` 具备整批边界；固定数值、定长数值、一层变长数值分支及自动计数
分支同生共死；失败后 `num_entries`、`num_baskets`、各分支读回内容、文件空闲段与目录
登记都对应成功前缀；关闭重开一致；故障清除后原对象可继续追加，顺序与变长值归属不变；
空批与输入校验失败不推进边界。

不保证：恢复过程本身再次失败、进程被杀、断电等多故障/崩溃场景（需求明确排除）。
失败可能留下未引用的文件空洞，不要求逐字节恢复成故障前的文件。
