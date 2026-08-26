import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from modulo_vki import ModuloVKI
from modulo_vki.test.test_modulo import synthetic_dataset


class TestFastMPOD(unittest.TestCase):
    def setUp(self):
        self.fs = 128.0
        self.data = synthetic_dataset(fs=self.fs)
        self.kwargs = dict(F_V=[12.0], fs=self.fs, ncpus=1, GThresh=0)

    def run_mode(self, mode, **kwargs):
        model = ModuloVKI(self.data, n_Modes=4, dtype="float64")
        with contextlib.redirect_stdout(io.StringIO()):
            return model.fastmPOD(mode=mode, **self.kwargs, **kwargs)

    def test_all_python_backends_agree(self):
        results = {
            mode: self.run_mode(mode)
            for mode in ("fullK", "bandK", "fullSVD", "randSVD")
        }
        phi_ref, psi_ref, sigma_ref = results["fullK"]
        reconstruction = phi_ref @ np.diag(sigma_ref) @ psi_ref.T
        relative_error = np.linalg.norm(self.data - reconstruction) / np.linalg.norm(
            self.data
        )
        self.assertLess(relative_error, 1e-5)
        np.testing.assert_allclose(psi_ref.T @ psi_ref, np.eye(4), atol=1e-10)

        projector_ref = psi_ref @ psi_ref.T
        for mode, (_, psi, sigma) in results.items():
            tolerance = 2e-4 if mode == "randSVD" else 1e-10
            np.testing.assert_allclose(sigma, sigma_ref, rtol=tolerance, atol=1e-8)
            np.testing.assert_allclose(
                psi @ psi.T, projector_ref, rtol=tolerance, atol=tolerance
            )

    def test_fortran_is_opt_in_and_missing_extension_is_actionable(self):
        with mock.patch(
            "modulo_vki.fortran.load_sym_matmul",
            side_effect=AssertionError("Fortran loader called unexpectedly"),
        ):
            self.run_mode("fullK", useFortran=False)

        with mock.patch(
            "modulo_vki.fortran.load_sym_matmul",
            side_effect=ModuleNotFoundError("optional extension missing"),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "optional extension missing"):
                self.run_mode("fullK", useFortran=True)

    def test_invalid_inputs_fail_early(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.run_mode("fullK", Keep=[0, 0])
        with self.assertRaisesRegex(ValueError, "only in fullK and bandK"):
            self.run_mode("fullSVD", useFortran=True)
        weighted = ModuloVKI(
            self.data,
            n_Modes=4,
            dtype="float64",
            weights=np.ones(self.data.shape[0]),
        )
        with self.assertRaisesRegex(NotImplementedError, "Weighted"):
            weighted.fastmPOD(**self.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
