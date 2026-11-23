"""Independent fault injector + checker (standalone, uses only public API).

The injector wraps a seekable, readable, writable file-like object at the
exact interface uproot.sink.file.FileSink consumes (seek/read/write/flush/
tell/close), so injected faults line up one-to-one with the sink's write and
flush calls -- independent of any buffering below.
"""
from __future__ import annotations

import io
import os

import numpy as np
import awkward as ak

import uproot


MODES = ("raise_before", "partial", "short", "zero", "flush")


class FaultyFile:
    """Seekable file-like object with a one-shot write/flush fault.

    Modes:
      raise_before: write raises OSError before touching any byte
      partial:      write overwrites part of the request, then raises OSError
      short:        write returns a positive count smaller than requested
      zero:         write returns 0
      flush:        flush raises OSError once
    """

    def __init__(self, name, mode, fail_on_call):
        assert mode in MODES
        self._fh = open(name, "w+b")
        self.mode = mode
        self.fail_on_call = fail_on_call
        self.calls = []  # (kind, position, length)
        self._armed = False
        self._arm_at = 0
        self._fired = False

    def arm(self):
        # fail_on_call is interpreted relative to arming time
        self._armed = True
        self._arm_at = len(self.calls)

    @property
    def fired(self):
        return self._fired

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
        if self._armed and not self._fired and idx == self.fail_on_call:
            self._fired = True
            if self.mode == "raise_before":
                raise OSError("injected write failure (no bytes written)")
            if self.mode == "partial":
                k = max(1, len(b) // 3)
                self._fh.write(b[:k])
                raise OSError("injected write failure (partial overwrite)")
            if self.mode == "short":
                self._fh.write(b[: max(1, len(b) // 2)])
                return max(1, len(b) // 2)
            if self.mode == "zero":
                return 0
        return self._fh.write(b)

    def flush(self):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("f", self._fh.tell(), 0))
        if (
            self._armed
            and not self._fired
            and self.mode == "flush"
            and idx == self.fail_on_call
        ):
            self._fired = True
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


class FaultyBytesIO:
    """Same fault semantics, backed purely by memory (file-like object case)."""

    def __init__(self, mode, fail_on_call):
        self._buf = io.BytesIO()
        self.mode = mode
        self.fail_on_call = fail_on_call
        self.calls = []
        self._armed = False
        self._arm_at = 0
        self._fired = False

    def arm(self):
        self._armed = True
        self._arm_at = len(self.calls)

    @property
    def fired(self):
        return self._fired

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
        if self._armed and not self._fired and idx == self.fail_on_call:
            self._fired = True
            if self.mode == "raise_before":
                raise OSError("injected write failure (no bytes written)")
            if self.mode == "partial":
                k = max(1, len(b) // 3)
                self._buf.write(b[:k])
                raise OSError("injected write failure (partial overwrite)")
            if self.mode == "short":
                self._buf.write(b[: max(1, len(b) // 2)])
                return max(1, len(b) // 2)
            if self.mode == "zero":
                return 0
        return self._buf.write(b)

    def flush(self):
        idx = len(self.calls) - self._arm_at
        self.calls.append(("f", self._buf.tell(), 0))
        if (
            self._armed
            and not self._fired
            and self.mode == "flush"
            and idx == self.fail_on_call
        ):
            self._fired = True
            raise OSError("injected flush failure")

    def close(self):
        # Keep the in-memory buffer readable after the writer closes its
        # handle, mirroring how an on-disk file stays readable on close.
        try:
            self._buf.flush()
        except ValueError:
            pass

    @property
    def closed(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def batch(seed):
    rng = np.random.default_rng(1000 + seed)
    return {
        "x": np.array([100 + 2 * seed, 100 + 2 * seed + 1], dtype=np.int64),
        "hits": ak.Array(
            [rng.normal(size=int(rng.integers(0, 4))).tolist(), []]
        ),
    }


def expected(nbatches):
    xs, hits = [], []
    for s in range(nbatches):
        b = batch(s)
        xs.append(b["x"])
        hits.append(b["hits"])
    return np.concatenate(xs), ak.concatenate(hits)


def check_prefix(handle_or_path, nbatches, tree_name="events", extras=None, seeds=None):
    """Open an independent read handle and verify the committed prefix."""
    if seeds is None:
        seeds = list(range(nbatches))
    with uproot.open(handle_or_path) as ro:
        t = ro[tree_name]
        assert t.num_entries == 2 * nbatches, (t.num_entries, nbatches)
        assert t["x"].num_baskets == nbatches
        xs, hits = [], []
        for s in seeds:
            b = batch(s)
            xs.append(b["x"])
            hits.append(b["hits"])
        got = t.arrays(["x", "hits"])
        assert np.array_equal(got["x"].to_numpy(), np.concatenate(xs)), got["x"]
        assert ak.all(got["hits"] == ak.concatenate(hits)), got["hits"]
        for bname in ("x", "nhits", "hits"):
            br = t[bname]
            n = br.num_baskets
            assert n == nbatches
            be = br.member("fBasketEntry")[: n + 1]
            assert int(be[0]) == 0 and int(be[-1]) == 2 * nbatches
            assert [int(be[i + 1] - be[i]) for i in range(n)] == [2] * n
            assert int(br.member("fWriteBasket")) == nbatches
            assert int(br.member("fEntries")) == 2 * nbatches
        if extras:
            for name, value in extras.items():
                assert str(ro[name]) == value
    return True


def run_case(tmpdir, mode, call_index, n_before=4, backend="file"):
    if backend == "file":
        path = os.path.join(
            tmpdir, f"case_{mode}_{call_index}.root"
        )
        faulty = FaultyFile(path, mode, call_index)
        reopen_path = path
    elif backend == "bytesio":
        path = None
        faulty = FaultyBytesIO(mode, call_index)
        # BytesIO keeps no on-disk path; after close, snapshot its bytes for reads
        reopen_path = None
    else:
        raise ValueError(backend)

    expected_kind = "f" if mode == "flush" else "w"

    with uproot.recreate(faulty) as out:
        tree = out.mktree(
            "events",
            {"x": np.int64, "hits": "var * float64"},
            initial_basket_capacity=3,
            resize_factor=10.0,
        )
        for s in range(n_before):
            tree.extend(batch(s))
        prefix_entries = tree.num_entries
        prefix_baskets = tree.num_baskets

        faulty.arm()
        calls_at_arm = len(faulty.calls)
        raised = False
        try:
            tree.extend(batch(n_before))
        except OSError:
            raised = True

        # raise_before/partial/zero/flush faults must surface; a one-shot short
        # write may instead be transparently completed by the sink's retry loop.
        if mode != "short":
            assert raised, f"{mode}@{call_index}: expected OSError"
        assert faulty.fired, (
            f"{mode}@{call_index}: fault never fired; call at index was "
            f"{faulty.calls[calls_at_arm + call_index]}"
        )
        # the fault must have hit the intended kind of operation
        fired_call = faulty.calls[calls_at_arm + call_index]
        assert fired_call[0] == expected_kind, fired_call

        if raised:
            # in-memory boundary unchanged immediately after the failure
            assert tree.num_entries == prefix_entries
            assert tree.num_baskets == prefix_baskets
        else:
            # short write was retried to completion: whole batch committed once
            assert tree.num_entries == prefix_entries + 2
            assert tree.num_baskets == prefix_baskets + 1

        # independent read handle while the write handle is still open
        mid_batches = n_before if raised else n_before + 1
        mid_seeds = (
            list(range(n_before))
            if raised
            else list(range(n_before + 1))
        )
        if backend == "file":
            check_prefix(reopen_path, mid_batches, seeds=mid_seeds)
        else:
            p2 = os.path.join(tmpdir, "bytesio_check.root")
            with open(p2, "wb") as fh:
                fh.write(faulty._buf.getvalue())
            check_prefix(p2, mid_batches, seeds=mid_seeds)

        # the original WritableTree must accept a new batch after the fault
        tree.extend(batch(n_before + 100))
        if raised:
            assert tree.num_entries == prefix_entries + 2
            assert tree.num_baskets == prefix_baskets + 1
        else:
            # the faulted batch already committed via retry; this is another one
            assert tree.num_entries == prefix_entries + 4
            assert tree.num_baskets == prefix_baskets + 2

    # after closing the write handle, reopen and verify the full content
    if raised:
        final_batches = n_before + 1
        final_seed_list = list(range(n_before)) + [n_before + 100]
    else:
        final_batches = n_before + 2
        final_seed_list = list(range(n_before + 1)) + [n_before + 100]
    if backend == "file":
        check_prefix(reopen_path, final_batches, seeds=final_seed_list)
        with uproot.open(reopen_path) as ro:
            t = ro["events"]
            xs, hits = [], []
            for s in final_seed_list:
                b = batch(s)
                xs.append(b["x"])
                hits.append(b["hits"])
            got = t.arrays(["x", "hits"])
            assert np.array_equal(got["x"].to_numpy(), np.concatenate(xs))
            assert ak.all(got["hits"] == ak.concatenate(hits))
    else:
        p2 = os.path.join(tmpdir, "bytesio_check2.root")
        with open(p2, "wb") as fh:
            fh.write(faulty._buf.getvalue())
        check_prefix(p2, final_batches, seeds=final_seed_list)
    return path
