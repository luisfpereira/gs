"""Execution backends.

Lead authors: Johan Mathe and Niklas Koep.
"""

import importlib
import logging
import os
import sys
import types

import gs._common as common

__version__ = "0.1.0"


def get_backend_name():
    return os.environ.get("GEOMSTATS_BACKEND", "numpy")


BACKEND_NAME = get_backend_name()

# --- Merge attributes from both backends ---
BACKEND_ATTRIBUTES = {
    "": [
        # Types
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "uint8",
        # Functions
        "abs",
        "all",
        "allclose",
        "amax",
        "amin",
        "angle",
        "any",
        "arange",
        "arccos",
        "arccosh",
        "arcsin",
        "arctan2",
        "arctanh",
        "argmax",
        "argmin",
        "array",
        "array_from_sparse",
        "asarray",
        "as_dtype",
        "assignment",
        "assignment_by_sum",
        "atol",
        "broadcast_arrays",
        "broadcast_to",
        "cast",
        "ceil",
        "clip",
        "comb",
        "concatenate",
        "conj",
        "convert_to_wider_dtype",
        "copy",
        "cos",
        "cosh",
        "cross",
        "cumprod",
        "cumsum",
        "diag_indices",
        "diagonal",
        "divide",
        "dot",
        "einsum",
        "empty",
        "empty_like",
        "equal",
        "erf",
        "exp",
        "expand_dims",
        "eye",
        "flatten",
        "flip",
        "floor",
        "from_numpy",
        "gamma",
        "get_default_dtype",
        "get_default_cdtype",
        "get_slice",
        "greater",
        "has_autodiff",
        "hsplit",
        "hstack",
        "imag",
        "isclose",
        "isnan",
        "is_array",
        "is_complex",
        "is_floating",
        "is_bool",
        "kron",
        "less",
        "less_equal",
        "linspace",
        "log",
        "logical_and",
        "logical_or",
        "mat_from_diag_triu_tril",
        "matmul",
        "matvec",
        "maximum",
        "mean",
        "meshgrid",
        "minimum",
        "mod",
        "moveaxis",
        "ndim",
        "one_hot",
        "ones",
        "ones_like",
        "outer",
        "pad",
        "pi",
        "polygamma",
        "power",
        "prod",
        "quantile",
        "ravel_tril_indices",
        "real",
        "repeat",
        "reshape",
        "rtol",
        "scatter_add",
        "searchsorted",
        "set_default_dtype",
        "set_diag",
        "shape",
        "sign",
        "sin",
        "sinh",
        "split",
        "sqrt",
        "squeeze",
        "sort",
        "stack",
        "std",
        "sum",
        "take",
        "tan",
        "tanh",
        "tile",
        "to_numpy",
        "to_ndarray",
        "trace",
        "transpose",
        "tril",
        "triu",
        "tril_indices",
        "triu_indices",
        "tril_to_vec",
        "triu_to_vec",
        "vec_to_diag",
        "unique",
        "vectorize",
        "vstack",
        "where",
        "zeros",
        "zeros_like",
        "trapezoid",
        # --- New backend attributes ---
        "geomspace",
        "scatter_sum_1d",
        "square",
        "argsort",
        "to_torch",
        "diag",
        "to_device",
    ],
    "autodiff": [
        "custom_gradient",
        "hessian",
        "hessian_vec",
        "jacobian",
        "jacobian_vec",
        "jacobian_and_hessian",
        "value_and_grad",
        "value_and_jacobian",
        "value_jacobian_and_hessian",
    ],
    "linalg": [
        "cholesky",
        "det",
        "eig",
        "eigh",
        "eigvalsh",
        "expm",
        "fractional_matrix_power",
        "inv",
        "is_single_matrix_pd",
        "logm",
        "matrix_power",
        "norm",
        "qr",
        "quadratic_assignment",
        "polar",
        "solve",
        "solve_sylvester",
        "sqrtm",
        "svd",
        "matrix_rank",
    ],
    "random": [
        "choice",
        "normal",
        "multivariate_normal",
        "rand",
        "randint",
        "seed",
        "uniform",
    ],
    "sparse": [
        "to_dense",
        "from_scipy_coo",
        "from_scipy_csc",
        "from_scipy_csr",
        "from_scipy_dia",
        "to_scipy_csc",
        "to_scipy_dia",
        "csr_matrix",
        "csc_matrix",
        "coo_matrix",
        "dia_matrix",
        "to_torch_csc",
        "to_torch_dia",
        "to_torch_coo",
        "to_coo",
        "to_csc",
        "to_csr",
    ],
}


class BackendImporter:
    """Importer class to create the backend module."""

    def __init__(self, path):
        self._path = self.name = path
        self.loader = self

    @staticmethod
    def _import_backend(backend_name):
        try:
            return importlib.import_module(f"gs.{backend_name}")
        except ModuleNotFoundError:
            try:
                return importlib.import_module(f"gs._backend.{backend_name}")
            except ModuleNotFoundError:
                raise RuntimeError(f"Unknown backend '{backend_name}'")

    def _create_backend_module(self, backend_name):
        backend = self._import_backend(backend_name)

        new_module = types.ModuleType(self._path)
        new_module.__file__ = getattr(backend, "__file__", None)

        for module_name, attributes in BACKEND_ATTRIBUTES.items():
            if module_name:
                try:
                    submodule = getattr(backend, module_name)
                except AttributeError:
                    raise RuntimeError(
                        f"Backend '{backend_name}' exposes no '{module_name}' module"
                    ) from None
                new_submodule = types.ModuleType(f"{self._path}.{module_name}")
                new_submodule.__file__ = getattr(submodule, "__file__", None)
                setattr(new_module, module_name, new_submodule)
            else:
                submodule = backend
                new_submodule = new_module
            for attribute_name in attributes:
                try:
                    submodule_ = submodule
                    if module_name == "" and not hasattr(submodule, attribute_name):
                        submodule_ = common
                    attribute = getattr(submodule_, attribute_name)
                except AttributeError:
                    if module_name:
                        error = (
                            f"Module '{module_name}' of backend '{backend_name}' "
                            f"has no attribute '{attribute_name}'"
                        )
                    else:
                        error = (
                            f"Backend '{backend_name}' has no "
                            f"attribute '{attribute_name}'"
                        )
                    raise RuntimeError(error) from None
                else:
                    setattr(new_submodule, attribute_name, attribute)

        return new_module

    def find_module(self, fullname, path=None):
        """Find module."""
        if self._path != fullname:
            return None
        return self

    def load_module(self, fullname):
        """Load module."""
        if fullname in sys.modules:
            return sys.modules[fullname]

        module = self._create_backend_module(BACKEND_NAME)
        module.__name__ = f"gs.{BACKEND_NAME}"
        module.__loader__ = self
        sys.modules[fullname] = module

        # Only set dtype if available
        if hasattr(module, "set_default_dtype"):
            module.set_default_dtype("float64")

        logging.debug(f"geomstats is using {BACKEND_NAME} backend")
        return module

    def find_spec(self, fullname, path=None, target=None):
        """Find module."""
        return self.find_module(fullname, path=path)


sys.meta_path.append(BackendImporter("gs.backend"))
