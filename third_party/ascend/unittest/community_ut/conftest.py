import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import npu_compat  # noqa: F401,E402


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "backend(name): select the runtime backend for a test module")


@pytest.fixture
def device():
    return "npu"


@pytest.fixture
def fresh_triton_cache():
    yield from npu_compat._fresh_triton_cache()


@pytest.fixture
def fresh_knobs():
    yield from npu_compat._fresh_knobs()


@pytest.fixture
def fresh_knobs_except_libraries():
    yield from npu_compat._fresh_knobs_except_libraries()


@pytest.fixture
def with_allocator():
    yield from npu_compat._with_allocator()


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
