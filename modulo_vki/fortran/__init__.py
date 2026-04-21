from importlib import import_module


def load_sym_matmul():
    """Load the compiled symmetric matrix multiplication extension."""
    module_name = f"{__name__}.symMatmulRoutines"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                "Fortran support requested, but the compiled extension "
                "'modulo_vki.fortran.symMatmulRoutines' was not found. "
                "Build it with: python -m modulo_vki.fortran.compile_fortran"
            ) from exc
        raise
