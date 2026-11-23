"""Minimal reproducer for the WritableTree.extend recovery bug.

Scenario from the issue:
  * mktree with an int64 branch ``x`` and a one-level var * float64 branch
    ``hits`` (plus its auto counter ``nhits``), initial_basket_capacity=3;
  * two batches commit successfully;
  * the third batch hits a one-shot write failure.

On the unpatched writer the exception escapes while num_entries /
num_baskets and branch basket tables have already advanced, so the on-disk
tree advertises baskets that do not exist. On the patched writer the failed
batch is fully retracted: the boundary stays at the previous prefix, an
independent read handle agrees, and the same WritableTree appends again once
the one-shot fault is spent.

Run with the project virtualenv, isolated mode:

    /path/to/venv/bin/python -I repro.py
"""
from __future__ import annotations

import os
import tempfile

import awkward as ak
import numpy as np

import uproot


class FaultyFile:
    """Seekable file-like object; its first armed write raises OSError once."""

    def __init__(self, name):
        self._fh = open(name, "w+b")
        self.armed = False
        self.fired = False

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
        if self.armed and not self.fired:
            self.fired = True
            raise OSError("injected one-shot write failure")
        return self._fh.write(b)

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()

    @property
    def closed(self):
        return self._fh.closed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def batch(seed):
    return {
        "x": np.array([10 * seed, 10 * seed + 1], dtype=np.int64),
        "hits": ak.Array([[float(seed), 2.0], []]),
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "events.root")
        faulty = FaultyFile(path)

        with uproot.recreate(faulty) as out:
            tree = out.mktree(
                "events",
                {"x": np.int64, "hits": "var * float64"},
                initial_basket_capacity=3,
            )
            tree.extend(batch(0))
            tree.extend(batch(1))
            prefix = (tree.num_entries, tree.num_baskets)
            print("committed prefix:", prefix)

            faulty.armed = True
            try:
                tree.extend(batch(2))
            except OSError as err:
                print("extend raised as expected:", err)
            else:
                raise SystemExit("fault did not fire")

            assert faulty.fired
            print("boundary after failure:", (tree.num_entries, tree.num_baskets))
            assert (tree.num_entries, tree.num_baskets) == prefix, (
                "FAILED: the failed batch advanced the commit boundary"
            )

            # same object appends again once the (one-shot) fault is spent
            tree.extend(batch(2))
            assert (tree.num_entries, tree.num_baskets) == (
                prefix[0] + 2,
                prefix[1] + 1,
            )

        with uproot.open(path) as ro:
            reread = ro["events"]
            arrays = reread.arrays(["x", "hits"])
            assert reread.num_entries == 6
            assert arrays["x"].tolist() == [0, 1, 10, 11, 20, 21]
            assert arrays["hits"].tolist() == [[0.0, 2.0], [], [1.0, 2.0], [], [2.0, 2.0], []]
            for name in ("x", "nhits", "hits"):
                branch = reread[name]
                fbe = branch.member("fBasketEntry")[: branch.num_baskets + 1]
                assert int(fbe[0]) == 0 and int(fbe[-1]) == 6
        print("independent reopen: all three batches read back correctly")


if __name__ == "__main__":
    main()
