# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone runner for traced-kernel examples under ``third_party/spyre/example/``.

Deliberately independent of ``test/conftest.py``'s fixture-discovery
machinery: examples here are extracted verbatim from real torch-spyre
Inductor traces (see each kernel's own docstring), grouped by source model,
rather than hand-authored to exercise a specific structural variant like
``test/fixtures/``. Keeping the two runners separate means this directory
can grow independently as more traces are added, without perturbing the
existing structural test suite.

Quick-reference
---------------
- :data:`EXAMPLES`              — registry of example kernels discovered
                                  from ``example/**/meta.py``
- :class:`KTIRStructuralTester` — EXAMPLE-based setup + structural assertions
- :class:`KTIRCpuTester`        — extends with numerical CPU execution

Shared, pytest-independent machinery (``OpInfo``, ``walk_module``,
``make_ktir_mod``, ``StructuralAssertions``, ``compile_to_ttir``) is reused
directly from ``test/utils.py`` rather than duplicated — see that module's
own docstring for why it's kept free of pytest/conftest coupling.
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

_triton_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_triton_root / "third_party" / "spyre"))

_EXAMPLE_DIR = Path(__file__).parent

# Load test/utils.py by file path rather than adding third_party/spyre/test
# to sys.path and doing `from utils import ...`: that directory also holds
# test/conftest.py, and a bare top-level "conftest" module name would then
# resolve ambiguously between the two conftest.py files depending on
# sys.path order — this file's own EXAMPLES could silently end up shadowed
# by test/'s fixtures-based registry. File-path loading sidesteps the
# ambiguity entirely.
_utils_path = _triton_root / "third_party" / "spyre" / "test" / "utils.py"
_utils_spec = importlib.util.spec_from_file_location(
    "_spyre_example_test_utils", _utils_path,
)
_utils = importlib.util.module_from_spec(_utils_spec)
sys.modules["_spyre_example_test_utils"] = _utils
_utils_spec.loader.exec_module(_utils)

OpInfo = _utils.OpInfo
StructuralAssertions = _utils.StructuralAssertions
compile_to_ttir = _utils.compile_to_ttir
make_ktir_mod = _utils.make_ktir_mod
walk_module = _utils.walk_module


# ---------------------------------------------------------------------------
# EXAMPLES — registry of example kernels used by tests
#
# Populated by discovery over example/**/meta.py: each meta.py exports a
# ``VARIANTS`` dict that gets expanded into one entry per variant, plus any
# reference/oracle helpers used by those variants. Example folders may nest
# arbitrarily deep (grouped by traced model name); the registry key is the
# meta.py's directory path relative to example/, as a POSIX string (e.g.
# "Meta-Llama-3.1-8B-Instruct/torch.add.1_spyre").
#
# Per-variant entry shape (whichever fields the variant supplies):
#   kernel_fn     : @triton.jit function compiled on demand
#   signature     : dict[str, str] — runtime ABI types
#   constexprs    : dict[str, scalar] — compile-time constants
#   grid          : list[int] — per-axis hardware partition; one entry
#                   per tl.program_id axis read by the kernel. Passed
#                   to DistributeWork via SpyreOptions. Defaults to the
#                   backend's (32,) when omitted.
#   reference     : (inputs) -> np.ndarray — NumPy oracle
#   inputs        : lambda c: {"argN": np.array, ...}
#   output_key    : which inputs key holds the output buffer
#   func_name     : KTIR function name (defaults to kernel_fn.__name__)
#   parallel      : bool, default True — set False for single-program
#                   kernels with no tl.program_id (skips the whole
#                   test_work_distribution check).
#   distribution_loop : bool, default True — set False when grid ==
#                   xnumel (one program per unit of work): DistributeWork
#                   still emits ktdp.get_compute_tile_id, but produces no
#                   residual scf.for since there's nothing left to loop
#                   over per program. Narrower than "parallel": False —
#                   compute_tile_id is still asserted present.
#   extra_checks  : optional (tester) -> None for variant-specific asserts
#   xfail_numerical : optional str | dict for the numerical-test xfail mark
# ---------------------------------------------------------------------------


def _sanitize_pkg_segment(name: str) -> str:
    """Turn a directory name into a valid dotted-module-name segment.

    Example folders are grouped by traced model name (e.g.
    ``Meta-Llama-3.1-8B-Instruct``), which contains dots/hyphens that would
    otherwise split into extra (bogus) package levels or produce invalid
    identifiers when used verbatim in a dotted module name. The real
    filesystem path is unaffected — only the ``sys.modules`` key / dotted
    name is sanitized.
    """
    return re.sub(r"\W", "_", name)


def _ensure_pkg(name: str, pkg_dir: Path) -> None:
    """Register ``name`` in ``sys.modules`` as a package rooted at ``pkg_dir``.

    Deliberately does NOT require ``pkg_dir / "__init__.py"`` to exist on
    disk for every level: the top-level "example" package in particular
    must NOT have a real ``__init__.py`` at ``example/``, since its mere
    presence would make pytest treat ``example/`` as an import package,
    changing ``test_examples.py``'s module name to ``example.test_examples``
    and breaking its plain ``from conftest import ...`` (which relies on
    ``example/`` being pytest's insertion-root, exactly like ``test/``).
    Falls back to a namespace-style module (``__path__`` set, no source)
    when no ``__init__.py`` is present.
    """
    if name in sys.modules:
        return
    init_file = pkg_dir / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(
            name, init_file, submodule_search_locations=[str(pkg_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        if spec.loader is not None:
            spec.loader.exec_module(mod)
    else:
        mod = types.ModuleType(name)
        mod.__path__ = [str(pkg_dir)]
        sys.modules[name] = mod


def _import_meta(meta_path: Path):
    """Import ``example/<...>/meta.py`` as a package-qualified module.

    The meta.py uses ``from . import kernel``, so we must import it as a
    member of the ``example[.<...>]`` package — bootstrap each parent
    package first, walking arbitrarily many intermediate directories (an
    example may be nested, e.g. grouped under a traced model name).
    """
    rel_parts = meta_path.parent.relative_to(_EXAMPLE_DIR).parts

    pkg_dir = _EXAMPLE_DIR
    pkg_name = "example"
    _ensure_pkg(pkg_name, pkg_dir)
    for part in rel_parts:
        pkg_dir = pkg_dir / part
        pkg_name = f"{pkg_name}.{_sanitize_pkg_segment(part)}"
        _ensure_pkg(pkg_name, pkg_dir)

    mod_name = f"{pkg_name}.meta"
    spec = importlib.util.spec_from_file_location(mod_name, meta_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_variant(
    module_sig: dict, entry: dict, *, kernel_name: str,
) -> tuple[dict, dict, dict]:
    """Resolve a variant's ``(runtime_signature, constexprs, param_values)``.

    Inputs:
      - ``module_sig`` — module-level ``SIGNATURE`` dict (maps arg name
        to dtype string). Used only if the variant doesn't declare its
        own ``SIGNATURE``.
      - ``entry`` — the fully-merged variant dict. Must declare
        ``constexpr`` (list[str]) and ``params`` (dict[str, list[Any]]).
        May declare ``SIGNATURE`` to override the module-level default
        wholesale (use when the variant's kernel has a different arg
        list).

    Outputs:
      - ``runtime_signature`` — ``{name: dtype}`` subset of the effective
        ``SIGNATURE`` for arg names not in the variant's ``constexpr``.
      - ``constexprs`` — ``{name: value}`` for arg names in ``constexpr``.
        The value is ``params[name][0]`` (Cartesian expansion is deferred).
      - ``param_values`` — ``{name: value}`` flattened from ``params``
        using ``[0]``. Used by the numerical test to build runtime kwargs.
    """
    if "params" not in entry:
        raise ValueError(f"{kernel_name}: variant missing 'params'")
    if "constexpr" not in entry:
        raise ValueError(f"{kernel_name}: variant missing 'constexpr'")

    param_values: dict = {}
    for pname, values in entry["params"].items():
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(
                f"{kernel_name}: params[{pname!r}] must be a 1-element list "
                f"(Cartesian expansion deferred); got {values!r}"
            )
        param_values[pname] = values[0]

    effective_sig = entry.get("SIGNATURE", module_sig)
    constexpr_names = set(entry["constexpr"])
    runtime_signature = {
        name: dtype for name, dtype in effective_sig.items()
        if name not in constexpr_names
    }
    constexprs = {name: param_values[name] for name in constexpr_names}
    return runtime_signature, constexprs, param_values


def _load_examples():
    registry: dict = {}
    if not _EXAMPLE_DIR.exists():
        return registry
    # Only folders with a meta.py qualify as kernel examples. Nesting is
    # arbitrary depth (examples are grouped under a traced model name).
    for meta_path in sorted(_EXAMPLE_DIR.glob("**/meta.py")):
        name = meta_path.parent.relative_to(_EXAMPLE_DIR).as_posix()
        mod = _import_meta(meta_path)
        module_sig = getattr(mod, "SIGNATURE", {})
        variants = mod.VARIANTS
        default = variants["default"]
        for vname, delta in variants.items():
            # Shallow merge: variant dict overrides default wholesale per key.
            merged = {**default, **delta}
            if module_sig:
                runtime, constexprs, param_values = _resolve_variant(
                    module_sig, merged, kernel_name=f"{name}::{vname}"
                )
                merged["signature"] = runtime
                merged["constexprs"] = constexprs
                merged["param_values"] = param_values
            key = name if vname == "default" else f"{name}__{vname}"
            registry[key] = merged
    return registry


EXAMPLES = _load_examples()


# ---------------------------------------------------------------------------
# KTIRStructuralTester — EXAMPLE-based setup + structural assertions
# ---------------------------------------------------------------------------

class KTIRStructuralTester(StructuralAssertions):
    """Pytest base class for KTIR structural checks on example kernels.

    Subclass, set ``EXAMPLE`` to a key in :data:`EXAMPLES`, and write test
    methods using the assertion helpers inherited from
    :class:`StructuralAssertions`. ``setup_method`` builds ``self.ops``
    automatically.
    """

    EXAMPLE: str = None

    def setup_method(self):
        if self.EXAMPLE is None:
            self._def_map = None
            return
        entry = EXAMPLES[self.EXAMPLE]
        grid = entry.get("grid")  # None → backend default

        if "kernel_fn" in entry:
            ttir_text = compile_to_ttir(
                entry["kernel_fn"],
                entry["signature"],
                entry.get("constexprs", {}),
            )
            import tempfile
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".mlir", delete_on_close=False) as f:
                f.write(ttir_text)
                f.flush()
                self.mod = make_ktir_mod(f.name, grid=grid)
        else:
            self.mod = make_ktir_mod(entry["path"], grid=grid)

        self.ops = walk_module(self.mod)
        self._def_map = None


# ---------------------------------------------------------------------------
# KTIRCpuTester — adds numerical CPU execution on top of structural checks
# ---------------------------------------------------------------------------

def _parse_mlir_frontend(mlir_text: str):
    """Parse via MLIRFrontendParser in a subprocess; return an IRModule.

    Why a subprocess: ``libtriton.so`` and ``mlir_ktdp`` each statically
    embed their own copy of MLIR, so co-loading both and parsing in-process
    segfaults from duplicate MLIR global state. See
    ``test/conftest.py``'s copy of this helper for the full rationale.
    """
    import pickle
    import subprocess
    worker = (
        "import sys, pickle\n"
        "from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser\n"
        "text = sys.stdin.buffer.read().decode()\n"
        "module = MLIRFrontendParser().parse_module(text)\n"
        "sys.stdout.buffer.write(pickle.dumps(module))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", worker],
        input=mlir_text.encode(),
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode()
        if "mlir_ktdp" in stderr or "tools_ktdp" in stderr:
            hint = (
                "MLIRFrontendParser (mlir_ktdp) is not installed; install the "
                "bindings from the ktir-mlir-frontend submodule by running "
                "'bash install-ktdp-mlir-bindings.sh' (generated by setup.py "
                "during the editable install)."
            )
        else:
            hint = "MLIRFrontendParser raised an error (mlir_ktdp is installed)."
        raise RuntimeError(
            f"MLIR parse worker failed. {hint}"
            f"\n--- worker stderr ---\n{stderr}"
        )
    return pickle.loads(proc.stdout)


class KTIRCpuTester:
    """Mixin that adds numerical CPU execution via ``ktir_cpu``.

    Designed for multiple inheritance with :class:`KTIRStructuralTester`.
    Requires ``self.mod`` (a live ``ir.module``) to be set by
    ``setup_method``. If ``ktir_cpu`` is not installed, :meth:`run_cpu`
    raises ``pytest.skip``.
    """

    def run_cpu(self, func_name: str, *, kernel_fn, **kwargs):
        """Execute ``self.mod`` via the ``ktir_cpu`` interpreter.

        See ``test/conftest.py``'s copy of this method for the full
        rationale behind the source-name ↔ argN remapping.
        """
        import pytest
        import inspect
        try:
            from ktir_cpu import KTIRInterpreter
        except ImportError:
            pytest.skip("ktir_cpu not installed — skipping numerical check")

        try:
            module = _parse_mlir_frontend(str(self.mod))
        except RuntimeError as e:
            pytest.fail(str(e))
        interp = KTIRInterpreter()
        interp.module = module

        raw_fn = getattr(kernel_fn, "fn", kernel_fn)
        source_names = list(inspect.signature(raw_fn).parameters)
        declared = interp.module.get_function(func_name).arg_names

        runtime_names = source_names[:len(declared)]
        name_to_arg = dict(zip(runtime_names, declared))

        unknown = set(kwargs) - set(runtime_names)
        if unknown:
            raise ValueError(
                f"run_cpu: unknown kwargs {sorted(unknown)} — declared "
                f"runtime args are {runtime_names}"
            )
        missing = set(runtime_names) - set(kwargs)
        if missing:
            raise ValueError(
                f"run_cpu: missing kwargs {sorted(missing)} — declared "
                f"runtime args are {runtime_names}"
            )

        argN_kwargs = {name_to_arg[k]: v for k, v in kwargs.items()}
        raw_outputs = interp.execute_function(func_name, **argN_kwargs)

        arg_to_name = {v: k for k, v in name_to_arg.items()}
        return {arg_to_name[k]: v for k, v in raw_outputs.items() if k in arg_to_name}
