"""Compare numerical outputs produced by two MODULO versions.

The command exits non-zero for missing files/keys, shape mismatches, non-finite
values, or tolerance violations, so it can be used as a release gate.
"""

import argparse
import os
import sys

import numpy as np


def align_columns(reference, candidate):
    """Align the real sign or complex phase of candidate columns."""
    aligned = candidate.astype(
        np.result_type(reference.dtype, candidate.dtype), copy=True
    )
    for column in range(reference.shape[1]):
        dot = np.vdot(reference[:, column], aligned[:, column])
        if np.abs(dot) > np.finfo(float).eps:
            aligned[:, column] *= np.conj(dot) / np.abs(dot)
    return aligned


def npz_paths(base):
    return {
        os.path.relpath(os.path.join(root, name), base)
        for root, _, files in os.walk(base)
        for name in files
        if name.endswith(".npz")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two directories of MODULO .npz regression outputs."
    )
    parser.add_argument("dir1", help="Reference-version output directory")
    parser.add_argument("dir2", help="Candidate-version output directory")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--allow-extra-candidate",
        action="store_true",
        help="Allow outputs present only in the candidate (for new features).",
    )
    args = parser.parse_args(argv)

    base1 = os.path.abspath(args.dir1)
    base2 = os.path.abspath(args.dir2)
    if not os.path.isdir(base1) or not os.path.isdir(base2):
        parser.error("both comparison inputs must be directories")

    paths1 = npz_paths(base1)
    paths2 = npz_paths(base2)
    failures = 0

    for relative in sorted(paths1 | paths2):
        print("\nFile: %s" % relative)
        if relative not in paths1:
            if args.allow_extra_candidate:
                print("  OK: new candidate-only output")
            else:
                print("  FAIL: missing from reference directory")
                failures += 1
            continue
        if relative not in paths2:
            print("  FAIL: missing from candidate directory")
            failures += 1
            continue

        with np.load(os.path.join(base1, relative)) as data1, np.load(
            os.path.join(base2, relative)
        ) as data2:
            for key in sorted(set(data1.files) | set(data2.files)):
                if key not in data1.files or key not in data2.files:
                    print("  FAIL %s: key missing from one output" % key)
                    failures += 1
                    continue
                reference = data1[key]
                candidate = data2[key]
                if reference.shape != candidate.shape:
                    print(
                        "  FAIL %s: shape %s != %s"
                        % (key, reference.shape, candidate.shape)
                    )
                    failures += 1
                    continue
                if not np.all(np.isfinite(reference)) or not np.all(
                    np.isfinite(candidate)
                ):
                    print("  FAIL %s: output contains NaN or infinity" % key)
                    failures += 1
                    continue

                if reference.ndim == 2 and reference.shape[1]:
                    candidate = align_columns(reference, candidate)
                close = np.allclose(
                    reference,
                    candidate,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                max_difference = (
                    float(np.max(np.abs(reference - candidate)))
                    if reference.size
                    else 0.0
                )
                status = "OK" if close else "FAIL"
                print("  %s %s: max abs diff %.3e" % (status, key, max_difference))
                failures += int(not close)

    if failures:
        print("\nComparison failed with %d issue(s)." % failures)
        return 1
    print("\nComparison passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
