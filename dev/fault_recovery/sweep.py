"""Drive the injector across every write/flush position of representative extends."""
from __future__ import annotations

import sys
import tempfile

import numpy as np

import uproot

sys.path.insert(0, ".")
import injector


def probe_calls(n_before, backend="file"):
    """Return the (kind, position, length) list produced by one target extend."""
    with tempfile.TemporaryDirectory() as d:
        if backend == "file":
            f = injector.FaultyFile(d + "/probe.root", "raise_before", 10**9)
        else:
            f = injector.FaultyBytesIO("raise_before", 10**9)
        with uproot.recreate(f) as out:
            t = out.mktree(
                "events",
                {"x": np.int64, "hits": "var * float64"},
                initial_basket_capacity=3,
            )
            for s in range(n_before):
                t.extend(injector.batch(s))
            mark = len(f.calls)
            t.extend(injector.batch(n_before))
            return f.calls[mark:]


def main():
    failures = []
    scenarios = (("capacity", 4), ("normal", 5))
    for label, n_before in scenarios:
        calls = probe_calls(n_before)
        write_idx = [i for i, c in enumerate(calls) if c[0] == "w"]
        flush_idx = [i for i, c in enumerate(calls) if c[0] == "f"]
        print(
            f"{label}: {len(calls)} calls "
            f"({len(write_idx)} writes, {len(flush_idx)} flushes)"
        )

        with tempfile.TemporaryDirectory() as d:
            for mode in ("raise_before", "partial", "short", "zero"):
                for ci in write_idx:
                    try:
                        injector.run_case(d, mode, ci, n_before=n_before)
                    except Exception as exc:  # noqa: BLE001
                        failures.append((label, mode, ci, type(exc).__name__, str(exc)))
            for ci in flush_idx:
                try:
                    injector.run_case(d, "flush", ci, n_before=n_before)
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        (label, "flush", ci, type(exc).__name__, str(exc))
                    )
            # in-memory file-like backend, using its own call layout
            mem_calls = probe_calls(n_before, backend="bytesio")
            mem_write = [i for i, c in enumerate(mem_calls) if c[0] == "w"]
            mem_flush = [i for i, c in enumerate(mem_calls) if c[0] == "f"]
            mem_pick = (
                mem_write[0],
                mem_write[len(mem_write) // 2],
                mem_write[-1],
                mem_flush[0],
                mem_flush[-1],
            )
            for mode in injector.MODES:
                for ci in mem_pick:
                    if (mode == "flush") != (mem_calls[ci][0] == "f"):
                        continue
                    try:
                        injector.run_case(
                            d, mode, ci, n_before=n_before, backend="bytesio"
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            (label + "-bytesio", mode, ci, type(exc).__name__, str(exc))
                        )

    print(f"\n{len(failures)} failures")
    for item in failures:
        print(item)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
