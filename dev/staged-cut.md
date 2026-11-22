# Staged cut reading (`cut_mode="staged"`)

`TTree.arrays` / `TTree.iterate` (and the module-level `uproot.iterate`) accept two
new keyword arguments:

| name | default | meaning |
| --- | --- | --- |
| `cut_mode` | `"eager"` | `"eager"` reads every branch the cut and the output expressions need, then filters. `"staged"` first reads only the branches the cut depends on, evaluates the cut, and then reads payload `TBasket`s for the surviving entries (plus any extra baskets the budget allows). |
| `cut_extra_bytes` | `0` | In `"staged"` mode, a non-negative, non-bool integer: extra compressed bytes that may be spent reading payload `TBasket`s that contain no surviving entry. The budget is shared across all output branches and charged by on-disk compressed basket size. |

When `cut is None`, `"staged"` is an ordinary read. The default (`"eager"`) is
byte-for-byte the previous behavior.

## Example

```python
import uproot

# Read only "pt" (the cut column) first, then only the "mass" baskets that
# contain a surviving event. Zero extra budget.
arrays = tree.arrays(
    ["mass", "met"],
    cut="pt > 30",
    library="ak",
    cut_mode="staged",
    cut_extra_bytes=0,
)

# Let the reader also pull in up to ~1 MB of payload baskets that hold no
# surviving entry (e.g. to reduce the number of tiny requests on dense data).
arrays = tree.arrays(
    ["mass", "met"],
    cut="pt > 30",
    library="ak",
    cut_mode="staged",
    cut_extra_bytes=1_000_000,
)

# Iterating keeps the original input entry windows for step_size and Report,
# and yields a correctly-typed (possibly empty) result for every window.
for batch, report in tree.iterate(
    ["mass", "met"],
    cut="pt > 30",
    library="ak",
    step_size=10_000,
    cut_mode="staged",
    report=True,
):
    print(report.tree_entry_start, report.tree_entry_stop, len(batch))
```

## What is read

* **Cut dependencies** (including columns reached through multi-level
  `aliases`) are always read over the full requested entry range — this is the
  minimum needed to evaluate the selection.
* **Output-only branches** are read only from `TBasket`s that contain a
  surviving entry when the budget is `0`. Survivor positions are grouped back
  into the same events by the interpretation step (flat, fixed-length, and
  one-level jagged numerical data), so a partial value inside one event is
  never mistaken for another event.
* The cut branch's own payload is reused for the output when the cut column is
  also requested.
* A jagged payload basket that stores no per-entry offsets pulls its count
  branch; only count baskets overlapping the surviving payload baskets are
  read.

## Constraints

* `cut_mode` must be `"eager"` or `"staged"`; anything else raises `ValueError`.
* `cut_extra_bytes` must be a non-negative, non-bool integer; anything else
  raises `TypeError`.
* The cut must select whole events (a 1-D boolean/numeric result). A
  per-element (jagged) mask raises `ValueError`.
* `"staged"` is implemented for `PythonLanguage` with `library="ak"`. Other
  languages/libraries raise `NotImplementedError` *before* any data `TBasket`
  is requested. Branches that are not flat/fixed-length/one-level-jagged
  numerical also raise `NotImplementedError` before reading.
* Results (order, dtype, empty lists, fixed-length dimensions) match the eager
  reader, including when no entries survive (the output still has the correct
  type without reading all payloads).

## Read-planning trade-offs

* **Sparse selections** gather only the survivor baskets; the planner ranks
  interior gap baskets by distance to the nearest hit so a positive budget buys
  the most "contiguous" extra baskets first.
* **Dense selections** favour interior gap baskets in entry order, so a
  positive budget merges runs of baskets into fewer, larger request schedules.
* The budget is a single shared pool: one extra basket costs its compressed
  bytes regardless of how many output branches it touches, and a basket is
  never requested twice. Extra reads beyond the required set are charged to the
  same pool.
* Resident memory grows with the current cut data, the baskets being processed,
  and the final selected result — not with the whole unselected payload.

## How this was verified

* Generated local `TTree` files with deliberately misaligned basket seams
  between the cut branch and payload branches.
* Wrapped the file `Source` to record every requested byte range (metadata
  excluded).
* Compared staged results against the eager reader element-by-element for:
  sparse survivors with whole unselected payload blocks between them, all/none
  selections, empty variable-length values, truncated ranges, multi-level
  aliases, shared cut/output dependencies, consecutive different cuts, and the
  array cache both on and off. Iteration was checked window-by-window,
  including empty windows and `Report` ranges.
* With zero budget, asserted no unselected payload basket is requested; with a
  positive budget, checked the extra/re-read bytes against `cut_extra_bytes`.
