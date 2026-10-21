import gc
import tracemalloc
import pytest
import pathlib
import os

import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from npu_compat import is_npu


def test_metadata() -> None:

    used_hook = False

    def _launch_metadata(grid, kernel, args):
        ret = dict()
        ret["grid"] = grid
        ret["value"] = args["x"]
        return ret

    def hook(launch_metadata):
        nonlocal used_hook
        metadata = launch_metadata.get()
        assert metadata["grid"] == (1, 3, 2)
        assert metadata["value"] == 6
        used_hook = True

    @triton.jit(launch_metadata=_launch_metadata)
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.launch_enter_hook.add(hook)
    kernel[(1, 3, 2)](6)
    triton.knobs.runtime.launch_enter_hook.remove(hook)
    assert used_hook


def test_memory_leak() -> None:

    device = "npu"

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    tracemalloc.start()
    try:
        inp = torch.randn(10, device=device)
        out = torch.randn(10, device=device)
        kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        begin, _ = tracemalloc.get_traced_memory()
        for _ in range(100):
            kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        end, _ = tracemalloc.get_traced_memory()
        assert end - begin < 30000
    finally:
        tracemalloc.stop()


def test_load_hook() -> None:

    used_start_hook = False
    start_hash = None

    def hook_start(module, function, name, metadata_group, hash):
        nonlocal used_start_hook
        nonlocal start_hash
        start_hash = hash
        used_start_hook = True

    used_end_hook = False
    end_hash = None

    def hook_end(module, function, name, metadata_group, hash):
        nonlocal used_end_hook
        nonlocal end_hash
        end_hash = hash
        used_end_hook = True

    @triton.jit
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end)
    kernel[(1, 3, 2)](6)
    assert used_start_hook
    assert used_end_hook
    assert start_hash == end_hash
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end)


def test_multiple_hooks() -> None:

    start0 = False
    end0 = False
    start1 = False
    end1 = False

    def hook_start0(module, function, name, metadata_group, hash):
        nonlocal start0
        start0 = True

    def hook_end0(module, function, name, metadata_group, hash):
        nonlocal end0
        end0 = True

    def hook_start1(module, function, name, metadata_group, hash):
        nonlocal start1
        start1 = True

    def hook_end1(module, function, name, metadata_group, hash):
        nonlocal end1
        end1 = True

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end1)

    @triton.jit
    def kernel(x):
        pass

    kernel[(1, )](6)

    assert start0
    assert end0
    assert start1
    assert end1

    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end1)


@pytest.mark.interpreter
def test_pre_run_hooks():

    device = "npu"

    @triton.jit
    def add_kernel(a_ptr, n_elements: tl.constexpr):
        offsets = tl.arange(0, n_elements)
        a = tl.load(a_ptr + offsets)
        a += 2
        tl.store(a_ptr + offsets, a)

    def my_hook(*args, **kwargs):
        args[0].zero_()

    add_kernel.add_pre_run_hook(my_hook)

    n_elements = 4
    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel[(1, )](a, n_elements)
    assert torch.all(a == 2)

    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel.run(a, n_elements, grid=(1, ), warmup=False)
    assert torch.all(a == 2)
