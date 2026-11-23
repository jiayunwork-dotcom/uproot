# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Batch-level recovery tests for the ROOT in-place writer.

A single ``WritableTree.extend`` call must have an all-or-nothing boundary:
after a recoverable write/flush failure the file and the in-memory tree must
look exactly as they did before the call, and the same WritableTree must be
able to keep appending once the (one-shot) fault is cleared.

The fault injector and the verification routines deliberately use only the
public uproot API and independent file handles -- they never inspect the
writer's internal recovery flags or count captured exceptions.
"""
from __future__ import annotations

import io
import os

import awkward as ak
import numpy as np
import pytest

import uproot

WRITE_MODES = ("raise_before", "partial", "short", "zero")
ALL_MODES = (*WRITE_MODES, "flush")


class FaultyFile:
    """Seekable, readable, writable file-like object with a one-shot fault.

    ``fail_on_call`` counts write/flush calls relative to :meth:`arm`.
    Modes:
      raise_before: write raises OSError before touching any byte
      partial:      write overwrites part of the request, then raises OSError
      short:        write returns a positive count smaller than requested
      zero:         write returns 0
      flush:        flush raises OSError once
    """

    def __init__(self, name, mode, fail_on_call):
        assert mode in ALL_MODES
        self._fh = open(name, "w+b")
        self.mode = mode
        self.fail_on_call = fail_on_call
        self.calls = []
        self._armed = False
        self._arm_at = 0
        self.fired = False

    def arm(self):
        self._armed = True
        self._arm_at = len(self.calls)

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return True

    def seek(self, *args):
        return self._fh.seek(*args)

    def tell(self):
        return self._fh.tell()

    def read(self, n=-1):
        return self._fh.read(n)

    def write(self, b):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("w", self._fh.tell(), len(b)))
        if self._armed and not self.fired and idx == self.fail_on_call:
            self.fired = True
            if self.mode == "raise_before":
                raise OSError("injected write failure (no bytes written)")
            if self.mode == "partial":
                self._fh.write(b[: max(1, len(b) // 3)])
                raise OSError("injected write failure (partial overwrite)")
            if self.mode == "short":
                k = max(1, len(b) // 2)
                self._fh.write(b[:k])
                return k
            if self.mode == "zero":
                return 0
        return self._fh.write(b)

    def flush(self):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("f", self._fh.tell(), 0))
        if (
            self._armed
            and not self.fired
            and self.mode == "flush"
            and idx == self.fail_on_call
        ):
            self.fired = True
            raise OSError("injected flush failure")
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    @property
    def closed(self):
        return self._fh.closed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FaultyBytesIO(FaultyFile):
    """Same fault semantics, backed purely by memory."""

    def __init__(self, mode, fail_on_call):
        self._fh = None
        self._buf = io.BytesIO()
        self.mode = mode
        self.fail_on_call = fail_on_call
        self.calls = []
        self._armed = False
        self._arm_at = 0
        self.fired = False

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return True

    def seek(self, *args):
        return self._buf.seek(*args)

    def tell(self):
        return self._buf.tell()

    def read(self, n=-1):
        return self._buf.read(n)

    def write(self, b):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("w", self._buf.tell(), len(b)))
        if self._armed and not self.fired and idx == self.fail_on_call:
            self.fired = True
            if self.mode == "raise_before":
                raise OSError("injected write failure (no bytes written)")
            if self.mode == "partial":
                self._buf.write(b[: max(1, len(b) // 3)])
                raise OSError("injected write failure (partial overwrite)")
            if self.mode == "short":
                k = max(1, len(b) // 2)
                self._buf.write(b[:k])
                return k
            if self.mode == "zero":
                return 0
        return self._buf.write(b)

    def flush(self):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("f", self._buf.tell(), 0))
        if (
            self._armed
            and not self.fired
            and self.mode == "flush"
            and idx == self.fail_on_call
        ):
            self.fired = True
            raise OSError("injected flush failure")

    def close(self):
        # an in-memory buffer stays readable after the writer's handle closes
        try:
            self._buf.flush()
        except ValueError:
            pass

    @property
    def closed(self):
        return False

    def snapshot(self):
        return self._buf.getvalue()


def make_batch(seed):
    rng = np.random.default_rng(1000 + seed)
    return {
        "x": np.array([10 * seed, 10 * seed + 1], dtype=np.int64),
        # an empty list and a non-empty list inside every batch
        "hits": ak.Array([[], rng.normal(size=int(rng.integers(1, 4))).tolist()]),
    }


def build_expected(seeds):
    xs = np.concatenate([make_batch(s)["x"] for s in seeds])
    hits = ak.concatenate([make_batch(s)["hits"] for s in seeds])
    return xs, hits


def assert_prefix(path_or_bytes, nbatches, seeds, tree_name="events"):
    """Independent read handle: content, counts and per-branch ranges."""
    with uproot.open(path_or_bytes) as ro:
        tree = ro[tree_name]
        assert tree.num_entries == 2 * nbatches
        x_exp, hits_exp = build_expected(seeds)
        arrays = tree.arrays(["x", "hits"])
        assert np.array_equal(arrays["x"].to_numpy(), x_exp)
        assert ak.all(arrays["hits"] == hits_exp)
        # ordering/ownership of variable-length values
        assert arrays["hits"].tolist() == hits_exp.tolist()
        for branch_name in ("x", "nhits", "hits"):
            branch = tree[branch_name]
            n = branch.num_baskets
            assert n == nbatches
            fbe = branch.member("fBasketEntry")[: n + 1]
            assert int(fbe[0]) == 0
            assert int(fbe[-1]) == 2 * nbatches
            assert [int(fbe[i + 1] - fbe[i]) for i in range(n)] == [2] * n
            assert int(branch.member("fWriteBasket")) == n
            assert int(branch.member("fEntries")) == 2 * nbatches


def call_layout(faulty, n_before):
    """Record the (kind) sequence of one target extend on a healthy twin."""
    with uproot.recreate(faulty) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        for seed in range(n_before):
            tree.extend(make_batch(seed))
        mark = len(faulty.calls)
        tree.extend(make_batch(n_before))
    return [c[0] for c in faulty.calls[mark:]]


def _run_fault_case(tmp_path, mode, call_index, n_before, backend):
    if backend == "file":
        path = os.path.join(
            tmp_path, f"fault_{mode}_{call_index}_{n_before}.root"
        )
        faulty = FaultyFile(path, mode, call_index)
    else:
        path = None
        faulty = FaultyBytesIO(mode, call_index)

    with uproot.recreate(faulty) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        for seed in range(n_before):
            tree.extend(make_batch(seed))
        prefix_entries = tree.num_entries
        prefix_baskets = tree.num_baskets

        faulty.arm()
        raised = False
        try:
            tree.extend(make_batch(n_before))
        except OSError:
            raised = True

        if mode != "short":
            assert raised, f"{mode} fault was expected to surface"
        assert faulty.fired

        if raised:
            # the boundary stays exactly at the previously committed prefix
            assert tree.num_entries == prefix_entries
            assert tree.num_baskets == prefix_baskets
            mid_seeds = list(range(n_before))
        else:
            # a one-shot short write may be fully retried: one clean commit
            assert tree.num_entries == prefix_entries + 2
            assert tree.num_baskets == prefix_baskets + 1
            mid_seeds = list(range(n_before + 1))

        if backend == "file":
            assert_prefix(path, len(mid_seeds), mid_seeds)
        else:
            check_file = os.path.join(tmp_path, "mid_check.root")
            with open(check_file, "wb") as handle:
                handle.write(faulty.snapshot())
            assert_prefix(check_file, len(mid_seeds), mid_seeds)

        # same object keeps appending after the fault clears
        recovery_seed = n_before + 100
        tree.extend(make_batch(recovery_seed))
        if raised:
            assert tree.num_entries == prefix_entries + 2
            assert tree.num_baskets == prefix_baskets + 1
        else:
            # the faulted batch already committed once, so this is a second
            # additional batch
            assert tree.num_entries == prefix_entries + 4
            assert tree.num_baskets == prefix_baskets + 2

    final_seeds = [*mid_seeds, recovery_seed]
    if backend == "file":
        assert_prefix(path, len(final_seeds), final_seeds)
    else:
        check_file = os.path.join(tmp_path, "final_check.root")
        with open(check_file, "wb") as handle:
            handle.write(faulty.snapshot())
        assert_prefix(check_file, len(final_seeds), final_seeds)


@pytest.mark.parametrize("n_before", [2, 4])
@pytest.mark.parametrize("mode", ALL_MODES)
def test_fault_at_every_write_and_flush_position(tmp_path, n_before, mode):
    # A healthy twin establishes which call positions actually exist in this
    # extend; each position gets an independent run with an independent fault.
    probe = FaultyFile(
        os.path.join(tmp_path, f"probe_{mode}_{n_before}.root"),
        mode,
        10**9,
    )
    kinds = call_layout(probe, n_before)
    probe.close()

    target_kind = "f" if mode == "flush" else "w"
    for index, kind in enumerate(kinds):
        if kind != target_kind:
            continue
        _run_fault_case(tmp_path, mode, index, n_before, backend="file")


def test_faults_through_in_memory_filelike(tmp_path):
    kinds = call_layout(FaultyBytesIO("raise_before", 10**9), 4)
    write_positions = [i for i, k in enumerate(kinds) if k == "w"]
    flush_positions = [i for i, k in enumerate(kinds) if k == "f"]
    picks = (
        write_positions[0],
        write_positions[len(write_positions) // 2],
        write_positions[-1],
    )
    for mode in WRITE_MODES:
        for index in picks:
            _run_fault_case(tmp_path, mode, index, 4, backend="bytesio")
    for index in (flush_positions[0], flush_positions[-1]):
        _run_fault_case(tmp_path, "flush", index, 4, backend="bytesio")


def test_two_metadata_expansions_recover_and_reread(tmp_path):
    """Six batches with capacity 3 / resize 2 => two TTree metadata moves."""
    path = os.path.join(tmp_path, "expansions.root")
    n = 6
    # healthy reference for comparison
    reference = os.path.join(tmp_path, "reference.root")
    with uproot.recreate(reference, compression=None) as out:
        ref = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        for seed in range(n):
            ref.extend(make_batch(seed))

    faulty = FaultyFile(path, "raise_before", 0)
    with uproot.recreate(faulty, compression=None) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        for seed in range(n - 1):
            tree.extend(make_batch(seed))
        before = (tree.num_entries, tree.num_baskets)

        # the fault lands inside the second metadata-expansion extend
        faulty.arm()
        with pytest.raises(OSError, match="injected"):
            tree.extend(make_batch(n - 1))
        assert (tree.num_entries, tree.num_baskets) == before
        assert_prefix(path, n - 1, list(range(n - 1)))

        tree.extend(make_batch(n - 1))
        tree.extend(make_batch(n + 50))

    assert_prefix(path, n + 1, [*range(n), n + 50])
    with uproot.open(path) as ro, uproot.open(reference) as ref_ro:
        # the healthy prefix matches a file that never saw a fault
        assert ro["events"].num_entries == ref_ro["events"].num_entries + 2
        assert (
            ro["events"]["x"].num_baskets
            == ref_ro["events"]["x"].num_baskets + 1
        )


def test_empty_and_invalid_batches_do_not_advance_boundary(tmp_path):
    path = os.path.join(tmp_path, "boundary.root")
    with uproot.recreate(path) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
        )
        tree.extend(make_batch(0))
        entries = tree.num_entries
        baskets = tree.num_baskets

        # empty batch: no basket, no allocation, no boundary movement
        tree.extend(
            {"x": np.array([], dtype=np.int64), "hits": ak.Array([[], []])[:0]}
        )
        assert tree.num_entries == entries
        assert tree.num_baskets == baskets

        # validation failures must likewise leave the boundary untouched
        with pytest.raises(ValueError, match="must be given an array for every branch"):
            tree.extend({"x": np.array([1, 2], dtype=np.int64)})
        with pytest.raises(ValueError, match="same number of entries"):
            tree.extend(
                {
                    "x": np.array([1], dtype=np.int64),
                    "hits": ak.Array([[1.0], [2.0]]),
                }
            )
        with pytest.raises(TypeError, match="requires a mapping"):
            tree.extend([1, 2, 3])
        assert tree.num_entries == entries
        assert tree.num_baskets == baskets

        # and the tree still accepts a real batch
        tree.extend(make_batch(1))
    assert_prefix(path, 2, [0, 1])


def test_three_interleaved_objects_survive_repeated_failures(tmp_path):
    """A second tree and a plain object must not be corrupted by tree failures."""
    path = os.path.join(tmp_path, "interleaved.root")
    faulty = FaultyFile(path, "raise_before", 10**9)
    with uproot.recreate(faulty) as out:
        events = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        other = out.mktree("other", {"y": np.float64}, initial_basket_capacity=3)
        events.extend(make_batch(0))
        events.extend(make_batch(1))
        other.extend({"y": np.array([0.5, 1.5])})
        out["label"] = "hello"

        # Repeatedly fail the FIRST write of events.extend: that position
        # deterministically exists in every extend, so no twin is needed.
        # Healthy writes to the other two objects are interleaved each round.
        for round_index in range(3):
            faulty.mode = "partial"
            faulty.fail_on_call = 0
            faulty.fired = False
            faulty.arm()
            with pytest.raises(OSError, match="injected"):
                events.extend(make_batch(2))
            assert faulty.fired
            assert events.num_entries == 4
            other.extend({"y": np.array([2.5, 3.5])})
            out[f"note_{round_index}"] = f"world_{round_index}"

        # recovery: the failed batch can finally be committed on a clean run
        faulty.mode = "raise_before"
        faulty.fail_on_call = 10**9
        faulty.fired = True  # disarm injection
        events.extend(make_batch(2))
        assert events.num_entries == 6

    with uproot.open(path) as ro:
        assert ro.keys(cycle=False) == [
            "events",
            "other",
            "label",
            "note_0",
            "note_1",
            "note_2",
        ]
        assert str(ro["label"]) == "hello"
        assert str(ro["note_2"]) == "world_2"
        # other tree accumulated three healthy recovery rounds
        assert ro["other"].arrays()["y"].tolist() == [
            0.5, 1.5, 2.5, 3.5, 2.5, 3.5, 2.5, 3.5
        ]
        assert_prefix(path, 3, [0, 1, 2])


def test_failure_during_expansion_with_tight_directory(tmp_path):
    """A capacity move may also reallocate the directory data block."""
    path = os.path.join(tmp_path, "tight_directory.root")
    faulty = FaultyFile(path, "partial", 1)
    with uproot.recreate(faulty, initial_directory_bytes=64) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=2.0,
        )
        for seed in range(2):
            tree.extend(make_batch(seed))
        # third extend reaches the basket-capacity boundary and moves the
        # TTree metadata; the fault fires inside the move
        before = (tree.num_entries, tree.num_baskets)
        faulty.arm()
        with pytest.raises(OSError, match="injected"):
            tree.extend(make_batch(2))
        assert faulty.fired
        assert (tree.num_entries, tree.num_baskets) == before

        # directory must still enumerate the tree and point at valid old data
        assert "events" in out

        # same object retries successfully once the one-shot fault is spent
        tree.extend(make_batch(2))
        assert (tree.num_entries, tree.num_baskets) == (before[0] + 2, before[1] + 1)

    assert_prefix(path, 3, [0, 1, 2])


def test_successful_extends_cover_compression_and_branch_layouts(tmp_path):
    """Success-path coverage: compression on/off and different branch order."""
    for compression in (None, uproot.ZLIB(1)):
        path = os.path.join(tmp_path, f"success_{compression}.root")
        with uproot.recreate(path, compression=compression) as out:
            tree = out.mktree(
                "events",
                # var-length branch first, fixed numeric branch after it
                {"hits": "var * float64", "x": np.int64},
                initial_basket_capacity=3,
                resize_factor=2.0,
            )
            for seed in range(6):
                tree.extend(make_batch(seed))
        assert_prefix(path, 6, list(range(6)))
        with uproot.open(path) as ro:
            branch = ro["events"]["hits"]
            assert branch.num_baskets == 6
            assert branch.member("fBasketEntry").tolist()[0] == 0
