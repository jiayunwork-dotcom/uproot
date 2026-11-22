# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests for staged ``cut`` reading (``cut_mode="staged"`` and ``cut_extra_bytes``).

These tests build their own local ``TTree`` files (so basket seams can be made
to disagree between the cut branch and the payload branches), wrap the file's
``Source`` to record which byte ranges are actually requested, and compare the
staged results with the ordinary ("eager") reader.
"""

import numpy
import pytest

ak = pytest.importorskip("awkward")

import uproot
import uproot.source.chunk


class TrackingSource(uproot.source.chunk.Source):
    """
    A ``Source`` that records every byte range it is asked for and looks up the
    last instance by file path.
    """

    registry = {}

    def __init__(self, path, **kwargs):
        super().__init__()
        self._file_path = path
        self._handle = open(path, "rb")
        self._handle.seek(0, 2)
        self._num_bytes = self._handle.tell()
        self.ranges = []
        TrackingSource.registry[path] = self

    def chunk(self, start, stop):
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start
        self.ranges.append((start, stop))
        self._handle.seek(start)
        data = self._handle.read(stop - start)
        return uproot.source.chunk.Chunk(
            self, start, stop, uproot.source.futures.TrivialFuture(data)
        )

    def chunks(self, ranges, notifications):
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)
        out = []
        for start, stop in ranges:
            self.ranges.append((start, stop))
            self._handle.seek(start)
            data = self._handle.read(stop - start)
            chunk = uproot.source.chunk.Chunk(
                self, start, stop, uproot.source.futures.TrivialFuture(data)
            )
            notifications.put(chunk)
            out.append(chunk)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._handle.close()

    def close(self):
        self._handle.close()

    @property
    def closed(self):
        return self._handle.closed


def _write_misaligned(path, spec, compression=None):
    """
    Writes a TTree where each branch can have its own ``TBasket`` block
    boundaries, so the cut branch and the payload branches have misaligned
    basket seams.

    ``spec`` maps branch name to ``(numpy dtype, 1-D array, blocks)`` where
    ``blocks`` is a list of ``(lo, hi)`` entry ranges, one ``TBasket`` per
    block. All branches must have the same number of blocks.
    """
    fout = uproot.recreate(path, compression=compression)
    try:
        fout.mktree("t", {name: dt for name, (dt, _array, _blocks) in spec.items()})
        tree = fout["t"]
        cascading = tree._cascading
        sink = fout._file.sink

        num_blocks = len(next(iter(spec.values()))[2])
        for _name, (_dt, _array, blocks) in spec.items():
            assert len(blocks) == num_blocks

        for block_index in range(num_blocks):
            for name, (_dt, array, blocks) in spec.items():
                datum = cascading._branch_data[cascading._branch_lookup[name]]
                lo, hi = blocks[block_index]
                sub = numpy.asarray(array[lo:hi]).astype(datum["dtype"])
                tot, zipb, location = cascading.write_np_basket(
                    sink, name, datum["compression"], sub
                )
                num = cascading._num_baskets
                datum["fBasketBytes"][num] = zipb
                if num + 1 < cascading._basket_capacity:
                    datum["fBasketEntry"][num + 1] = (
                        hi - lo
                    ) + datum["fBasketEntry"][num]
                datum["fBasketSeek"][num] = location
                datum["arrays_write_stop"] = num + 1
                datum["fTotBytes"] += tot
                datum["fZipBytes"] += zipb
            cascading._num_baskets += 1

        total = len(next(iter(spec.values()))[1])
        cascading._num_entries = total
        cascading._metadata["fEntries"] = total
        cascading.write_updates(sink)
        sink.flush()
    finally:
        fout.close()


def _basket_ranges(path):
    """Mapping ``(branch name, basket number) -> (byte start, byte stop)``."""
    out = {}
    with uproot.open(path) as file:
        tree = file["t"]
        for name in tree.keys():
            branch = tree[name]
            if branch.count_branch is not None:
                continue
            for basket_num in range(branch.num_baskets):
                start = int(branch.member("fBasketSeek")[basket_num])
                out[(name, basket_num)] = (
                    start,
                    start + branch.basket_compressed_bytes(basket_num),
                )
    return out


def _tracked(path, call):
    """Runs ``call(tree)`` with a tracking source and returns (result, ranges)."""
    file = uproot.open(path, handler=TrackingSource)
    source = TrackingSource.registry[path]
    source.ranges.clear()
    try:
        result = call(file["t"])
    finally:
        file.close()
    return result, list(source.ranges)


@pytest.fixture
def misaligned_file(tmp_path):
    path = str(tmp_path / "misaligned.root")
    rng = numpy.random.default_rng(0)
    n = 60
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    _write_misaligned(
        path,
        {
            "a": (numpy.float64, a, [(0, 20), (20, 40), (40, 60)]),
            "b": (numpy.float64, b, [(0, 5), (5, 35), (35, 60)]),
            "c": (numpy.float64, c, [(0, 33), (33, 47), (47, 60)]),
        },
        compression=uproot.ZLIB(1),
    )
    return path, {"a": a, "b": b, "c": c}


def _eager(path, exprs, cut, **kwargs):
    with uproot.open(path) as file:
        return file["t"].arrays(exprs, cut=cut, library="ak", **kwargs)


def _staged(path, exprs, cut, extra=0, **kwargs):
    result, _ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            exprs,
            cut=cut,
            library="ak",
            cut_mode="staged",
            cut_extra_bytes=extra,
            **kwargs,
        ),
    )
    return result


def test_staged_matches_eager_misaligned(misaligned_file):
    path, _ = misaligned_file
    for cut in ["a > 1.0", "a < -1.0", "a > 0.5", "a > 100", "a > -100"]:
        expected = _eager(path, ["a", "b", "c"], cut)
        actual = _staged(path, ["a", "b", "c"], cut)
        assert actual.layout.form == expected.layout.form
        assert ak.almost_equal(actual, expected), cut


def test_staged_zero_budget_only_reads_survivor_baskets(misaligned_file):
    path, arrays = misaligned_file
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    inverse = {value: key for key, value in basket_ranges.items()}

    # survivors are all in b's second basket and c's first basket
    survivors = numpy.nonzero(arrays["a"] > 1.5)[0]
    assert len(survivors) > 0

    result, ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            ["b", "c"], cut="a>1.5", library="ak", cut_mode="staged"
        ),
    )
    assert ak.almost_equal(result, _eager(path, ["b", "c"], "a>1.5"))

    requested = sorted(
        inverse[item] for item in ranges if item in payload
    )
    # "a" (the cut branch) is always read in full
    assert [item for item in requested if item[0] == "a"] == [
        ("a", 0),
        ("a", 1),
        ("a", 2),
    ]
    # only b/c baskets containing survivors were requested
    survivor_baskets = {
        (name, number)
        for name in ("b", "c")
        for number, (start, stop) in enumerate(
            _branch_entry_offsets(path, name)
        )
        if stop > survivors[0] and start < survivors[-1] + 1
    }
    for item in requested:
        if item[0] in ("b", "c"):
            assert item in survivor_baskets


def _branch_entry_offsets(path, name):
    with uproot.open(path) as file:
        branch = file["t"][name]
        offsets = branch.entry_offsets
    return list(zip(offsets[:-1], offsets[1:]))


def test_staged_sparse_gap_skips_between_baskets(misaligned_file):
    path, arrays = misaligned_file
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    inverse = {value: key for key, value in basket_ranges.items()}

    # Keep everything for b except the entries that fall in its zero-hit
    # middle region, so we can tell that a whole payload basket is skipped.
    # Entries 5..34 form b's second basket; make them never survive.
    # Simpler: choose a cut that survives only in entries 0..4 and 35..59.
    cut = "((a > 0) & (a < 0.5)) "
    result, ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            ["b"], cut=cut, library="ak", cut_mode="staged"
        ),
    )
    expected = _eager(path, ["b"], cut)
    assert ak.almost_equal(result, expected)

    b_requested = sorted(
        number
        for item in ranges
        if item in payload
        for name, number in [inverse[item]]
        if name == "b"
    )
    # entries surviving in b basket 1 must have requested it, but if no
    # survivor falls in it, it must not be requested.
    entry_offsets = _branch_entry_offsets(path, "b")
    a = arrays["a"]
    for number, (start, stop) in enumerate(entry_offsets):
        has_survivor = bool(
            numpy.any((a[start:stop] > 0) & (a[start:stop] < 0.5))
        )
        assert (number in b_requested) == has_survivor, (number, has_survivor)


def test_staged_empty_result_has_correct_type(misaligned_file):
    path, _ = misaligned_file
    expected = _eager(path, ["b", "c"], "a > 1000")
    result, ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            ["b", "c"], cut="a>1000", library="ak", cut_mode="staged"
        ),
    )
    assert len(result) == 0
    assert result.layout.form == expected.layout.form

    # Only the cut branch was read; no payload basket was requested.
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    inverse = {value: key for key, value in basket_ranges.items()}
    requested = [inverse[item] for item in ranges if item in payload]
    assert all(name == "a" for name, _number in requested)


def test_staged_extra_bytes_shared_budget(tmp_path):
    path = str(tmp_path / "gaps.root")
    rng = numpy.random.default_rng(7)
    n = 60
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    # Six small baskets per branch; the payload branches have the same seams so
    # the "interior gap" baskets are unambiguous.
    blocks = [(i, i + 10) for i in range(0, n, 10)]
    _write_misaligned(
        path,
        {
            "a": (numpy.float64, a, blocks),
            "b": (numpy.float64, b, blocks),
            "c": (numpy.float64, c, blocks),
        },
        compression=uproot.ZLIB(1),
    )

    # Survivors in baskets 0, 2, 4; baskets 1, 3, 5 have no survivors.
    chosen = [5, 25, 45]
    cut = "(" + ") | (".join(f"(a == {float(a[int(x)])!r})" for x in chosen) + ")"

    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    inverse = {value: key for key, value in basket_ranges.items()}
    sizes = {key: stop - start for key, (start, stop) in basket_ranges.items()}

    def requested_with(extra):
        _result, ranges = _tracked(
            path,
            lambda tree: tree.arrays(
                ["b", "c"],
                cut=cut,
                library="ak",
                cut_mode="staged",
                cut_extra_bytes=extra,
            ),
        )
        return sorted(inverse[item] for item in ranges if item in payload)

    zero = requested_with(0)
    required_b = {item for item in zero if item[0] == "b"}
    assert required_b == {("b", 0), ("b", 2), ("b", 4)}

    # Budget that fits one more basket buys exactly one extra basket overall,
    # not one per branch (the pool is shared).
    one_basket = min(sizes[("b", 1)], sizes[("b", 3)])
    with_one = requested_with(one_basket)
    extras = [item for item in with_one if item not in zero]
    assert len(extras) == 1, extras

    # A budget below any gap basket buys nothing extra.
    below = min(sizes[("b", 1)], sizes[("b", 3)]) - 1
    assert requested_with(below) == zero

    # A large budget fills all interior gaps (both b and c). The planner only
    # considers gaps *between* surviving baskets, so the trailing basket 5 is
    # not part of the optional set.
    with_all = requested_with(10**6)
    assert {item for item in with_all if item[0] in ("b", "c")} == {
        (name, number) for name in ("b", "c") for number in range(5)
    }

    # Correctness holds at every budget.
    expected = _eager(path, ["b", "c"], cut)
    for extra in [0, one_basket, 10**6]:
        assert ak.almost_equal(_staged(path, ["b", "c"], cut, extra=extra), expected)


def test_staged_truncated_range(misaligned_file):
    path, _ = misaligned_file
    for entry_start, entry_stop in [
        (2, 4),
        (3, 25),
        (19, 21),
        (0, 60),
        (20, 45),
        (59, 60),
        (10, 10),
    ]:
        expected = _eager(
            path,
            ["b", "c"],
            "a > 0",
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        result = _staged(
            path,
            ["b", "c"],
            "a > 0",
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        assert ak.almost_equal(result, expected), (entry_start, entry_stop)
        assert result.layout.form == expected.layout.form


def _stage_aliases(path):
    # Multi-level aliases: cut1 -> cut2 -> a, so the cut reaches "a" through
    # two aliases, and "b" is only needed for the output.
    return {"cut1": "cut2 + 0", "cut2": "a"}


def test_staged_multi_level_alias(misaligned_file):
    path, _ = misaligned_file
    aliases = _stage_aliases(path)
    expected = _eager(
        path, ["b", "c"], "cut1 > 1.0", aliases=aliases
    )
    result = _staged(path, ["b", "c"], "cut1 > 1.0", aliases=aliases)
    assert ak.almost_equal(result, expected)


def test_staged_shared_dependency(misaligned_file):
    path, _ = misaligned_file
    # "a" is both in the cut and the output; "a" must be read once (full).
    _result, ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            ["a * 2", "b"], cut="a>1.0", library="ak", cut_mode="staged"
        ),
    )
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    inverse = {value: key for key, value in basket_ranges.items()}
    requested = sorted(inverse[item] for item in ranges if item in payload)
    a_baskets = [item for item in requested if item[0] == "a"]
    assert len(a_baskets) == len(set(a_baskets))  # no duplicate reads
    assert ak.almost_equal(
        _staged(path, ["a * 2", "b"], "a>1.0"),
        _eager(path, ["a * 2", "b"], "a>1.0"),
    )


def test_staged_jagged_output(misaligned_file, tmp_path):
    path = str(tmp_path / "jagged.root")
    rng = numpy.random.default_rng(3)
    n = 60
    a = rng.normal(size=n)
    flat = rng.normal(size=n)
    # Build explicit jagged pieces per block.
    blocks = [(0, 20), (20, 40), (40, 60)]
    counts = [int(k) for k in rng.integers(0, 5, size=n)]
    starts = numpy.zeros(n + 1, dtype=numpy.int64)
    numpy.cumsum(counts, out=starts[1:])
    flat_values = rng.normal(size=int(starts[-1])).astype("f4")

    with uproot.recreate(path, compression=uproot.ZLIB(1)) as fout:
        fout.mktree(
            "t",
            {
                "a": numpy.float64,
                "f": numpy.float64,
                "v": ak.types.from_datashape("var * float32", highlevel=False),
            },
        )
        for lo, hi in blocks:
            sub_offsets = starts[lo : hi + 1] - starts[lo]
            arr = ak.Array(
                ak.contents.ListOffsetArray(
                    ak.index.Index64(sub_offsets),
                    ak.contents.NumpyArray(flat_values[starts[lo] : starts[hi]]),
                )
            )
            fout["t"].extend(
                {
                    "a": a[lo:hi],
                    "f": flat[lo:hi],
                    "v": arr,
                    "nv": counts[lo:hi],
                }
            )

    for cut in ["a > 0.5", "a < -0.5", "a > 100", "a > -100"]:
        expected = _eager(path, ["v", "f"], cut)
        result = _staged(path, ["v", "f"], cut)
        assert ak.almost_equal(result, expected), cut
        # The eager reader finalizes first and then applies a boolean mask,
        # which materializes a `ListArray`; the staged reader assembles
        # `ListOffsetArray`s directly. They are equivalent lists.
        assert result.type == expected.type, cut
        assert result["v"].to_list() == expected["v"].to_list()


def test_staged_jagged_mask_rejected(misaligned_file, tmp_path):
    path = str(tmp_path / "jagged2.root")
    rng = numpy.random.default_rng(4)
    n = 30
    with uproot.recreate(path) as fout:
        fout.mktree(
            "t",
            {
                "a": numpy.float64,
                "v": ak.types.from_datashape("var * float32", highlevel=False),
            },
        )
        v = ak.Array(
            [rng.normal(size=int(k)).astype("f4").tolist() for k in rng.integers(0, 4, size=n)]
        )
        fout["t"].extend(
            {
                "a": rng.normal(size=n),
                "v": v,
                "nv": numpy.asarray(ak.num(v), dtype=numpy.int32),
            }
        )

    with uproot.open(path) as file:
        with pytest.raises(ValueError):
            file["t"].arrays(["a"], cut="v > 0", library="ak", cut_mode="staged")


def test_staged_unsupported_language_library_before_reading(misaligned_file):
    path, _ = misaligned_file
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())

    # unsupported library
    with pytest.raises(NotImplementedError):
        file = uproot.open(path, handler=TrackingSource)
        source = TrackingSource.registry[path]
        source.ranges.clear()
        try:
            file["t"].arrays(
                ["b"], cut="a>1.0", library="np", cut_mode="staged"
            )
        finally:
            file.close()
    assert not any(item in payload for item in source.ranges)

    # unsupported language
    class OtherLanguage(uproot.language.Language):
        pass

    with pytest.raises(NotImplementedError):
        file = uproot.open(path, handler=TrackingSource)
        source = TrackingSource.registry[path]
        source.ranges.clear()
        try:
            file["t"].arrays(
                ["b"],
                cut="a>1.0",
                library="ak",
                language=OtherLanguage(),
                cut_mode="staged",
            )
        finally:
            file.close()
    assert not any(item in payload for item in source.ranges)


def test_staged_invalid_arguments(misaligned_file):
    path, _ = misaligned_file
    with uproot.open(path) as file:
        tree = file["t"]
        with pytest.raises(ValueError):
            tree.arrays(["b"], cut="a>1", cut_mode="bogus")
        for value in [True, False, -1, 1.5, "3", numpy.bool_(True)]:
            with pytest.raises(TypeError):
                tree.arrays(
                    ["b"], cut="a>1", cut_mode="staged", cut_extra_bytes=value
                )
        # a non-negative non-bool integer is accepted
        result = tree.arrays(
            ["b"], cut="a>1", cut_mode="staged", cut_extra_bytes=10
        )
        assert ak.almost_equal(result, tree.arrays(["b"], cut="a>1"))


def test_staged_no_cut_is_ordinary_read(misaligned_file):
    path, _ = misaligned_file
    expected = _eager(path, ["b", "c"], None)
    result = _staged(path, ["b", "c"], None)
    assert ak.almost_equal(result, expected)
    assert result.layout.form == expected.layout.form


def test_staged_iterate_windows_and_report(misaligned_file):
    path, _ = misaligned_file
    step = 7
    with uproot.open(path) as file:
        tree = file["t"]
        one = tree.arrays(["a", "b", "c"], cut="a>0.5", library="ak")
        staged = [
            (arrays, report)
            for arrays, report in tree.iterate(
                ["a", "b", "c"],
                cut="a>0.5",
                library="ak",
                step_size=step,
                cut_mode="staged",
                report=True,
            )
        ]
        eager = [
            (arrays, report)
            for arrays, report in tree.iterate(
                ["a", "b", "c"],
                cut="a>0.5",
                library="ak",
                step_size=step,
                report=True,
            )
        ]

    assert [
        (r.tree_entry_start, r.tree_entry_stop) for _, r in staged
    ] == [(r.tree_entry_start, r.tree_entry_stop) for _, r in eager]

    for (s_arrays, _), (e_arrays, _) in zip(staged, eager):
        assert s_arrays.layout.form == e_arrays.layout.form
        assert ak.almost_equal(s_arrays, e_arrays)

    assert ak.almost_equal(ak.concatenate([a for a, _ in staged]), one)


def test_staged_iterate_empty_window_has_correct_type(misaligned_file):
    path, _ = misaligned_file
    with uproot.open(path) as file:
        tree = file["t"]
        reference = tree.arrays(["b", "c"], cut="a>1000", library="ak")
        fragments = list(
            tree.iterate(
                ["b", "c"],
                cut="a>1000",
                library="ak",
                step_size=10,
                cut_mode="staged",
                report=True,
            )
        )
    assert reference.type == fragments[0][0].type
    assert len(fragments) == 6
    for arrays, report in fragments:
        assert len(arrays) == 0
        assert arrays.type == reference.type
        assert report.tree_entry_stop > report.tree_entry_start


def test_staged_consecutive_cuts_with_cache(misaligned_file):
    path, _ = misaligned_file
    cache = uproot.cache.LRUArrayCache("100 MB")
    for cut in ["a>1.0", "a<-1.0", "a>1.0", "a>0.5", "a>1.0"]:
        with uproot.open(path) as file:
            tree = file["t"]
            expected = tree.arrays(
                ["a", "b", "c"], cut=cut, library="ak", array_cache=cache
            )
            actual = tree.arrays(
                ["a", "b", "c"],
                cut=cut,
                library="ak",
                cut_mode="staged",
                array_cache=cache,
            )
        assert ak.almost_equal(actual, expected), cut
        assert actual.layout.form == expected.layout.form

    # Also without any cache.
    for cut in ["a>1.0", "a<-1.0"]:
        with uproot.open(path) as file:
            tree = file["t"]
            expected = tree.arrays(
                ["a", "b", "c"], cut=cut, library="ak", array_cache=None
            )
            actual = tree.arrays(
                ["a", "b", "c"],
                cut=cut,
                library="ak",
                cut_mode="staged",
                array_cache=None,
            )
        assert ak.almost_equal(actual, expected), cut


def test_staged_iterate_does_not_reread(misaligned_file):
    path, _ = misaligned_file
    file = uproot.open(path, handler=TrackingSource)
    source = TrackingSource.registry[path]
    source.ranges.clear()
    try:
        arrays = list(
            file["t"].iterate(
                ["a", "b", "c"],
                cut="a>1.0",
                library="ak",
                step_size=6,
                cut_mode="staged",
            )
        )
    finally:
        file.close()

    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())
    requested = [item for item in source.ranges if item in payload]
    # No basket is requested more than once across the whole iteration.
    assert len(requested) == len(set(requested))

    one = _eager(path, ["a", "b", "c"], "a>1.0")
    assert ak.almost_equal(ak.concatenate(arrays), one)


def test_staged_dense_selection_request_accounting(tmp_path):
    """
    A dense selection (most entries survive) spends *requests* on every payload
    basket it touches; a sparse one spends fewer. Both stay correct, and the
    requested-byte count never exceeds the full payload when the budget is
    zero.
    """
    path = str(tmp_path / "dense.root")
    rng = numpy.random.default_rng(11)
    n = 120
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    blocks = [(i, i + 20) for i in range(0, n, 20)]
    _write_misaligned(
        path,
        {
            "a": (numpy.float64, a, blocks),
            "b": (numpy.float64, b, blocks),
        },
        compression=uproot.ZLIB(1),
    )

    basket_ranges = _basket_ranges(path)
    payload_b = {
        (start, stop) for (name, _number), (start, stop) in basket_ranges.items()
        if name == "b"
    }
    total_b_payload = sum(stop - start for start, stop in payload_b)

    def payload_bytes(cut):
        _result, ranges = _tracked(
            path,
            lambda tree: tree.arrays(
                ["b"], cut=cut, library="ak", cut_mode="staged"
            ),
        )
        assert ak.almost_equal(_result, _eager(path, ["b"], cut))
        return sum(stop - start for start, stop in ranges if (start, stop) in payload_b)

    sparse = payload_bytes("a > 2.5")
    dense = payload_bytes("a > -10")

    assert sparse < total_b_payload
    assert dense == total_b_payload  # every b basket has a survivor


def test_staged_count_branch_dependency(tmp_path, monkeypatch):
    """
    If a jagged payload ``TBasket`` does not store its own entry offsets, the
    count branch must be read, and only the count ``TBaskets`` overlapping the
    surviving payload baskets should be requested.
    """
    path = str(tmp_path / "counts.root")
    rng = numpy.random.default_rng(13)
    n = 60
    counts = rng.integers(0, 4, size=n)
    a = rng.normal(size=n)
    blocks = [(i, i + 10) for i in range(0, n, 10)]
    with uproot.recreate(path, compression=uproot.ZLIB(1)) as fout:
        fout.mktree(
            "t",
            {
                "a": numpy.float64,
                "v": ak.types.from_datashape("var * float32", highlevel=False),
            },
        )
        for lo, hi in blocks:
            v = ak.Array(
                [
                    rng.normal(size=int(k)).astype("f4").tolist()
                    for k in counts[lo:hi]
                ]
            )
            fout["t"].extend(
                {
                    "a": a[lo:hi],
                    "v": v,
                    "nv": numpy.asarray(counts[lo:hi], dtype=numpy.int32),
                }
            )

    # Force the jagged basket to behave as if it had no per-entry offsets, so
    # that the count branch is the only source of lengths.
    from uproot.models import TBasket as _TBasket

    original = _TBasket.Model_TBasket.byte_offsets

    def missing_offsets(self):
        if self._parent.count_branch is not None and original.fget(self) is not None:
            return None
        return original.fget(self)

    monkeypatch.setattr(
        _TBasket.Model_TBasket, "byte_offsets", property(missing_offsets)
    )

    expected = _eager(path, ["v", "a"], "a > 0.5")
    result = _staged(path, ["v", "a"], "a > 0.5")
    assert ak.almost_equal(result, expected)
    assert result.type == expected.type


def test_staged_duplicate_reads_are_not_charged(tmp_path):
    """
    Within one ``arrays`` call, no payload basket is requested twice.
    """
    path = str(tmp_path / "dups.root")
    rng = numpy.random.default_rng(17)
    n = 60
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    blocks = [(i, i + 10) for i in range(0, n, 10)]
    _write_misaligned(
        path,
        {
            "a": (numpy.float64, a, blocks),
            "b": (numpy.float64, b, blocks),
            "c": (numpy.float64, c, blocks),
        },
        compression=uproot.ZLIB(1),
    )
    basket_ranges = _basket_ranges(path)
    payload = set(basket_ranges.values())

    _result, ranges = _tracked(
        path,
        lambda tree: tree.arrays(
            ["a", "b", "c"],
            cut="a>-10",
            library="ak",
            cut_mode="staged",
            cut_extra_bytes=10**6,
        ),
    )
    requested = [item for item in ranges if item in payload]
    assert len(requested) == len(set(requested))


def test_staged_fixed_length_and_jagged_mixed(tmp_path):
    """
    Fixed-length (per-entry shape) branches keep their inner dimensions; a
    jagged branch keeps its empty lists; all match the eager reader.
    """
    path = str(tmp_path / "mixed.root")
    rng = numpy.random.default_rng(19)
    n = 40
    a = rng.normal(size=n)
    fixed = rng.normal(size=(n, 3)).astype("f4")
    blocks = [(i, i + 10) for i in range(0, n, 10)]
    with uproot.recreate(path, compression=uproot.ZLIB(1)) as fout:
        fout.mktree(
            "t",
            {
                "a": numpy.float64,
                "fixed": numpy.dtype(("f4", (3,))),
                "v": ak.types.from_datashape("var * float64", highlevel=False),
            },
        )
        for lo, hi in blocks:
            v = ak.Array(
                [rng.normal(size=k).tolist() for k in rng.integers(0, 3, size=hi - lo)]
            )
            fout["t"].extend(
                {
                    "a": a[lo:hi],
                    "fixed": fixed[lo:hi],
                    "v": v,
                    "nv": numpy.asarray(ak.num(v), dtype=numpy.int32),
                }
            )

    for cut in ["a > 0", "a > 100", "a > -100", "a < -0.3"]:
        expected = _eager(path, ["fixed", "v", "a"], cut)
        result = _staged(path, ["fixed", "v", "a"], cut)
        assert ak.almost_equal(result, expected), cut
        assert result.type == expected.type
        assert numpy.asarray(result["fixed"]).shape[1:] == (3,)
        assert result["v"].to_list() == expected["v"].to_list()
