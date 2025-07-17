import argparse
import os
import numpy as np


def sign_align_2d(a, b):
    """
    Align sign/phase of each column of b to match a (both are 2D with same shape).
    """
    b_aligned = b.copy()
    for j in range(a.shape[1]):
        dp = np.vdot(a[:, j], b[:, j])
        if np.abs(dp) > 1e-12:
            phase = dp / np.abs(dp)
            b_aligned[:, j] *= phase
    return b_aligned


def main():
    parser = argparse.ArgumentParser(
        description='Simple comparison of two ModuloVKI output folders.'
    )
    parser.add_argument('dir1', help='First version output directory')
    parser.add_argument('dir2', help='Second version output directory')
    args = parser.parse_args()

    base1 = os.path.abspath(args.dir1)
    base2 = os.path.abspath(args.dir2)

    for root, _, files in os.walk(base1):
        rel = os.path.relpath(root, base1)
        for fname in sorted(files):
            if not fname.endswith('.npz'):
                continue
            path1 = os.path.join(root, fname)
            path2 = os.path.join(base2, rel, fname)

            print(f"\nFile: {rel}/{fname}")
            if not os.path.exists(path2):
                print(f"  MISSING in second directory: {path2}")
                continue

            d1 = np.load(path1)
            d2 = np.load(path2)
            keys = sorted(set(d1.files) | set(d2.files))

            for key in keys:
                print(f"  Key: '{key}'")
                if key not in d1.files:
                    print("    MISSING in first .npz")
                    continue
                if key not in d2.files:
                    print("    MISSING in second .npz")
                    continue

                a = d1[key]
                b = d2[key]
                # Compare shapes
                if a.shape == b.shape:
                    print(f"    Shapes: OK {a.shape}")
                else:
                    print(f"    Shapes: MISMATCH {a.shape} vs {b.shape}")
                    # cannot compare magnitudes if shapes differ
                    continue

                # Compare magnitudes
                # If 2D, align sign/phase first
                if a.ndim == 2:
                    b = sign_align_2d(a, b)
                diff = np.abs(a - b)
                max_diff = np.nanmax(diff)
                print(f"    Max abs diff: {max_diff:.3e}")

    print("\nComparison complete.")

if __name__ == '__main__':
    main()
