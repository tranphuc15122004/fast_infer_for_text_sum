"""Process-local compatibility helpers for this repo's FA4 b15/CUTLASS stack."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import sys
import types

def _install_flash_attention_4_cutlass_compat() -> bool:
    """Install the FA4 beta compatibility module without touching site-packages.

    ``flash-attn-4==4.0.0b15`` still imports the deprecated
    ``cutlass.utils.ampere_helpers`` module.  CUTLASS 4.5.2 removed that module
    but retained the only value used by the beta kernel: ``SMEM_CAPACITY``.
    Register the small compatibility module in ``sys.modules`` and adapt the
    legacy ``nvvm.fmax`` call convention for this process only, so the server
    environment and the FA2 installation remain unchanged.
    """

    changed = False
    module_name = "cutlass.utils.ampere_helpers"
    try:
        missing_ampere_helpers = importlib.util.find_spec(module_name) is None
    except Exception:
        missing_ampere_helpers = False

    if missing_ampere_helpers:
        try:
            cutlass_utils = importlib.import_module("cutlass.utils")
        except Exception:
            cutlass_utils = None
        if cutlass_utils is not None:
            shim = types.ModuleType(module_name)
            shim.SMEM_CAPACITY = {
                "sm80": 163840,
                "sm86": 102400,
                "sm89": 102400,
                "sm90": 232448,
                "sm100": 229376,
            }
            sys.modules[module_name] = shim
            setattr(cutlass_utils, "ampere_helpers", shim)
            changed = True

    # FA4 b15 calls nvvm.fmax directly with the optional third argument
    # positionally.  CUTLASS DSL 4.5.x made that argument keyword-only and
    # also made nvvm.fmax require MLIR Values rather than CuTe scalar objects.
    # Adapt the callable in this process instead of modifying either
    # site-packages tree.
    try:
        nvvm = importlib.import_module("cutlass._mlir.dialects.nvvm")
        fmax = nvvm.fmax
        cutlass = importlib.import_module("cutlass")
        float32 = cutlass.Float32
        mlir_ir = importlib.import_module("cutlass._mlir.ir")

        # CUTLASS only re-exports these NVVM enums from ``cute.arch`` for
        # CUDA 12.9.  FA4 b15 still imports them from that stable CuTe path,
        # while the server's CUDA 13 package keeps the same enums under
        # ``cutlass._mlir.dialects.nvvm``.  The CUDA 13 CuTe wrappers also
        # require string literals (for example ``"async.shared"``) rather
        # than enum instances.  Expose a small string-member namespace in
        # memory so FA4 can use the old member names with the new ABI.
        try:
            cute_arch = importlib.import_module("cutlass.cute.arch")
            for enum_name in (
                "ProxyKind",
                "SharedSpace",
                "RoundingModeKind",
                "ReduxKind",
                "AtomicOpKind",
            ):
                if not hasattr(cute_arch, enum_name) and hasattr(nvvm, enum_name):
                    enum_class = getattr(nvvm, enum_name)
                    string_members = {
                        member.name: str(member) for member in enum_class
                    }
                    setattr(
                        cute_arch,
                        enum_name,
                        types.SimpleNamespace(**string_members),
                    )
                    changed = True
        except Exception:
            # The fmax compatibility below remains independently useful for
            # CUTLASS builds that already expose the CuTe enum names.
            pass

        parameters = inspect.signature(fmax).parameters
        c_parameter = parameters.get("c")
        if (
            c_parameter is not None
            and c_parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not getattr(fmax, "_fast_infer_fa4_compat", False)
        ):
            @functools.wraps(fmax)
            def fmax_compat(a, b, *args, **kwargs):
                # FA4 b15 uses the pre-CUTLASS-4.5 signature:
                #   nvvm.fmax(T.f32(), a, b, c=...)
                # Newer CUTLASS infers the result type and accepts only
                #   nvvm.fmax(a, b, c=...)
                # Drop the explicit result type before adapting operands.
                result_type = None
                if isinstance(a, mlir_ir.Type):
                    result_type = a
                    operands = (b, *args)
                    if len(operands) < 2:
                        return fmax(a, b, *args, **kwargs)
                    a, b, *args = operands
                if len(args) == 1:
                    # FA4 b15 supplies c positionally; some generated DSL
                    # paths also leave a keyword ``c`` behind.  The
                    # positional value is the authoritative third operand.
                    kwargs["c"] = args[0]
                    args = ()
                if args:
                    return fmax(a, b, *args, **kwargs)
                loc = kwargs.get("loc")
                ip = kwargs.get("ip")
                c = kwargs.get("c")

                def as_ir_value(value):
                    ir_value = getattr(value, "ir_value", None)
                    if callable(ir_value):
                        return ir_value(loc=loc, ip=ip)
                    try:
                        return float32(value).ir_value(loc=loc, ip=ip)
                    except Exception as exc:
                        value_type = type(value)
                        value_attrs = tuple(
                            name
                            for name in (
                                "value",
                                "type",
                                "dtype",
                                "shape",
                                "__extract_mlir_values__",
                                "__new_from_mlir_values__",
                            )
                            if hasattr(value, name)
                        )
                        raise TypeError(
                            "FA4 nvvm.fmax operand is not directly convertible: "
                            f"type={value_type.__module__}.{value_type.__qualname__}, "
                            f"text={value!s}, attrs={value_attrs}"
                        ) from exc

                raw_kwargs = {
                    "a": as_ir_value(a),
                    "b": as_ir_value(b),
                    "c": as_ir_value(c) if c is not None else None,
                    "ftz": kwargs.get("ftz"),
                    "nan": kwargs.get("nan"),
                    "abs": kwargs.get("abs"),
                    "loc": loc,
                    "ip": ip,
                }
                raw_parameters = inspect.signature(fmax).parameters
                first_parameter = next(iter(raw_parameters), None)
                if first_parameter == "res":
                    # Some CUTLASS 4.5 builds retain the old explicit-result
                    # argument, while others infer it from a/b.
                    if result_type is None:
                        result_type = getattr(cutlass, "T", None)
                        result_type = (
                            result_type.f32()
                            if result_type is not None
                            else None
                        )
                    if result_type is None:
                        raise TypeError(
                            "CUTLASS nvvm.fmax requires `res`, but FA4 did not "
                            "provide an explicit result type"
                        )
                    raw_kwargs["res"] = result_type
                return fmax(**raw_kwargs)

            fmax_compat._fast_infer_fa4_compat = True
            nvvm.fmax = fmax_compat
            changed = True
    except Exception:
        # The import probe below remains authoritative for unsupported or
        # otherwise incomplete CUTLASS installations.
        pass

    return changed

def _probe_flash_attention_4() -> tuple[bool, str | None]:
    """Check that FA4 and its transitive CUDA/CuTe dependencies import.

    ``find_spec("flash_attn.cute")`` only proves that a package directory is
    discoverable.  FA4 loads CUTLASS/CuTe modules during import, so a stale
    ``flash_attn.cute`` tree can be discoverable while still being unusable.
    Keep the full exception as a preflight reason for actionable launcher
    output.
    """

    try:
        if importlib.util.find_spec("flash_attn.cute") is None:
            return False, "flash_attn.cute is not installed"
    except Exception as exc:  # discovery can import a broken parent package
        detail = str(exc).strip().splitlines()[0] or repr(exc)
        return False, f"{type(exc).__name__}: {detail}"

    _install_flash_attention_4_cutlass_compat()
    try:
        importlib.import_module("flash_attn.cute")
    except Exception as exc:
        detail = str(exc).strip().splitlines()[0] or repr(exc)
        return False, f"{type(exc).__name__}: {detail}"
    return True, None


def install_fa4_compat() -> bool:
    """Apply compatibility shims and verify that the FA4 import chain works."""
    changed = _install_flash_attention_4_cutlass_compat()
    available, reason = _probe_flash_attention_4()
    if not available:
        raise RuntimeError(f"FA4 CuTe runtime failed to import: {reason}")
    return changed
