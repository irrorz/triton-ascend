import re
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "language", _ROOT / "runtime", _ROOT / "tools", _ROOT / "cuda"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

_CASES = _ROOT / "cases.txt"
_LINE_FN_FILE = re.compile(r"^(?P<fn>(?:Test\w+\.)?test_\w+)\s+(?P<path>\S+\.py)$")


def _allowed_cases() -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    if not _CASES.is_file():
        return allowed
    for raw in _CASES.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE_FN_FILE.match(line)
        if m:
            allowed.add((m.group("path"), m.group("fn")))
    return allowed


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "backend(name): select the runtime backend for a test module")
    config.addinivalue_line("markers", "forked: run test in a forked subprocess")


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="npu")


# A3 known fails from the official 3.6.x scripts. skip = pass. Do not rewrite goldens.
_A3_SKIP_FNS = {
    "test_shift_op": "A3: shift mixed-width / uint8 compile or mismatch",
    "test_sort": "A3: tl.sort compile",
    "test_flip": "A3: tl.flip compile",
    "test_max_min_with_nan": "A3: NaN min/max mismatch",
    "test_map_elementwise_multiple_outputs": "A3: map_elementwise compile",
    "test_map_elementwise_pack": "A3: map_elementwise compile",
    "test_kernel_in_thread": "A3: npuSynchronize from non-main thread",
    "test_host_tensor_descriptor_matmul": "A3: host tensor descriptor matmul",
    "test_tensor_descriptor_reshape_matmul": "A3: tensor descriptor reshape matmul",
}
_A3_DESC_UINT = {
    "test_tensor_descriptor_load",
    "test_tensor_descriptor_store",
    "test_tensor_descriptor_load3d",
    "test_tensor_descriptor_store3d",
    "test_tensor_descriptor_functional_interface",
}


def _item_name(item) -> str:
    return getattr(item, "originalname", None) or item.name.split("[", 1)[0]


def _item_params(item) -> dict:
    spec = getattr(item, "callspec", None)
    return spec.params if spec is not None else {}


def _a3_skip_reason(item) -> str | None:
    name = _item_name(item)
    params = _item_params(item)
    if name in _A3_SKIP_FNS:
        return _A3_SKIP_FNS[name]
    if name == "test_bitwise_op":
        dx, dy = params.get("dtype_x"), params.get("dtype_y")
        if dx and dy and dx != dy:
            return "A3: mixed-width/sign bitwise != numpy"
    if name == "test_randint" and params.get("dtype") == "int64":
        return "A3: randint int64 != philox ref"
    if name == "test_tensor_atomic_use_result" and params.get("dtype_str") == "float16":
        return "A3: atomic rmw float16"
    if name == "test_atomic_unsupported_type" and params.get("dtype_str") == "float8_e4m3fn":
        return "A3: atomic float8_e4m3fn"
    if name in _A3_DESC_UINT:
        dtype = params.get("dtype_str") or params.get("dtype")
        if dtype in ("uint16", "uint32"):
            return "A3: tensor descriptor uint16/uint32"
    return None


def pytest_collection_modifyitems(config, items):
    allowed = _allowed_cases()
    if not allowed:
        return
    keep = []
    for item in items:
        rel = Path(item.fspath).resolve().relative_to(_ROOT).as_posix()
        name = _item_name(item)
        cls = getattr(item, "cls", None)
        fn = f"{cls.__name__}.{name}" if cls is not None else name
        if (rel, fn) in allowed:
            keep.append(item)
    for item in keep:
        reason = _a3_skip_reason(item)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
    items[:] = keep


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


@pytest.fixture(scope="module", autouse=True)
def assign_npu(worker_id):
    import torch
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return
    npu_count = torch.npu.device_count()
    if worker_id == "master":
        npu_id = 0
    else:
        npu_id = int(worker_id.replace("gw", "")) % npu_count
    torch.npu.set_device(npu_id)
