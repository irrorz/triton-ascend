"""NPU stand-in for community is_cuda() / CUDA SM queries.

torch_npu has no CUDA compute capability. Community tests index
get_device_capability()[0] (Hopper=9, Blackwell=10). On NPU that API
returns None and crashes. Treat NPU as Ampere (8, 0): Hopper/TMA/Blackwell
gates skip, Ampere-or-newer kernels still run.
"""
import torch

_NPU_FAKE_CC = (8, 0)


def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _looks_like_cc(cc) -> bool:
    return (isinstance(cc, tuple) and len(cc) >= 2 and isinstance(cc[0], int) and isinstance(cc[1], int))


def _install_cuda_cc_shim():
    if not hasattr(torch, "npu"):
        return
    orig = getattr(torch.npu, "get_device_capability", None)

    def get_device_capability(*args, **kwargs):
        if orig is not None:
            try:
                cc = orig(*args, **kwargs)
            except Exception:
                cc = None
            if _looks_like_cc(cc):
                return cc
        return _NPU_FAKE_CC

    torch.npu.get_device_capability = get_device_capability


_install_cuda_cc_shim()


def _fresh_triton_cache():
    import tempfile
    from triton import knobs

    with tempfile.TemporaryDirectory() as tmpdir:
        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


def _fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


def _fresh_knobs_except_libraries():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


def _with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")


def _device():
    return "npu"


def install_pytest_fixtures(ns):
    """Register conftest fixtures in a test module so subdirectory pytest works."""
    import pytest

    ns["device"] = pytest.fixture(_device)
    ns["fresh_triton_cache"] = pytest.fixture(_fresh_triton_cache)
    ns["fresh_knobs"] = pytest.fixture(_fresh_knobs)
    ns["fresh_knobs_except_libraries"] = pytest.fixture(_fresh_knobs_except_libraries)
    ns["with_allocator"] = pytest.fixture(_with_allocator)
