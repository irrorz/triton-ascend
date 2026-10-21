# ruff: noqa: F821,F841
import contextlib
import itertools
import re
from typing import Optional
import math
import textwrap

import numpy as np
import pytest
import torch
import inspect
from numpy.random import RandomState

import triton
import triton.language as tl

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from npu_compat import install_pytest_fixtures
install_pytest_fixtures(globals())
from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    str_to_triton_dtype,
    uint_dtypes,
    float_dtypes,
    float_dtypes_with_bfloat16,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hopper,
    is_hip,
    is_hip_cdna,
    is_hip_cdna2,
    is_hip_cdna3,
    is_hip_cdna4,
    is_hip_gfx11,
    is_hip_gfx12,
    is_xpu,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)
from triton.runtime.errors import InterpreterError


def _without_fp64(xs):
    return [x for x in xs if x not in ("float64", "f64")]


float_dtypes = _without_fp64(float_dtypes)
float_dtypes_with_bfloat16 = _without_fp64(float_dtypes_with_bfloat16)
dtypes = _without_fp64(dtypes)
dtypes_with_bfloat16 = _without_fp64(dtypes_with_bfloat16)
torch_dtypes = _without_fp64(list(torch_dtypes))


@contextlib.contextmanager
def promotion_numpy_2_0():
    state = np._get_promotion_state()
    np._set_promotion_state("weak")
    try:
        yield
    finally:
        np._set_promotion_state(state)


# No need to emulate NumPy 2.0 if the user has NumPy 2.0
if np.__version__[0] != "1":
    promotion_numpy_2_0 = contextlib.nullcontext

# TODO: enable multiple cta cluster testing.
# num_ctas_list = [1, 4] if torch.cuda.get_device_capability()[0] == 9 else [1]
num_ctas_list = [1]

mma_nonk_sizes = []

GPU_DIALECT = "ttg"
if is_interpreter():
    THREADS_PER_WARP = 1
elif is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
    # for CDNA multiple variants of mma instructions are supported:
    # mfma 16x16/mfma 32x32
    # 0 is a special value for automatic heuristic
    if is_hip_cdna():
        mma_nonk_sizes = [0, 16, 32]
    elif is_hip_gfx11() or is_hip_gfx12():
        mma_nonk_sizes = [16]
else:
    THREADS_PER_WARP = 32


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def _dtype(dtype: str) -> str:
    # ex.: "int64" -> "int"
    return re.match(r'([a-zA-Z]+)', dtype).group(0)


def patch_kernel(template, to_replace):
    if is_interpreter():
        local_namespace = {}
        src = textwrap.dedent(inspect.getsource(template.fn))
        for k, v in to_replace.items():
            src = src.replace(k, v)
        exec(src, globals(), local_namespace)
        return local_namespace[template.fn.__name__]
    else:
        kernel = triton.JITFunction(template.fn)
        src = kernel.src
        for key, value in to_replace.items():
            src = src.replace(key, value)
        kernel._unsafe_update_src(src)
        return kernel


def check_npu_or_hip(device):
    # NPU and HIP both use pytorch device 'npu' / 'hip'. Other backends like Intel
    # GPU do not.
    if device not in ['npu']:
        pytest.skip("Only for npu or HIP")


def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    # CUDA SM gates (bf16 needs sm80, e4m3fn needs sm90). npu_compat fakes
    # Ampere (8, 0) so those must not run on NPU — A5 does have e4m3.
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


def get_src_element_ty_size(dtype_str):
    if dtype_str in ["int8", "uint8", "float8e4b15"]:
        return 1
    if dtype_str == "float16":
        return 2
    if dtype_str == "float32" or dtype_str == "tensorfloat32":
        return 4
    if dtype_str == "float64":
        return 8
    raise ValueError(f"Unknown dtype {dtype_str}")


@pytest.mark.interpreter
def test_scalar_overflow():

    device = "npu"

    @triton.jit
    def kernel():
        huge_int: tl.constexpr = 0xFFFFFFFFFFFFFF
        x = tl.full((), 32, dtype=tl.int32)
        y = x + huge_int

    with pytest.raises(triton.TritonError, match="out of range"):
        kernel[(1, )]()


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='npu', num_ctas=1):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    # avoid log/sqrt of negative numbers
    if 'log' in expr or 'sqrt' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_x)
    kernel[(1, )](Z=z_tri, X=x_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='npu', num_ctas=1,
                 x_low=None, x_high=None, y_low=None, y_high=None, filter_y=None, test_broadcast=True,
                 test_scalar=True):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    check_type_supported(dtype_y, device)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_lhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_rhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_scalar_rhs(Z, X, y: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    replacements = {'GENERATE_TEST_HERE': expr}
    kernel = patch_kernel(kernel, replacements)
    kernel_broadcast_lhs = patch_kernel(kernel_broadcast_lhs, replacements)
    kernel_broadcast_rhs = patch_kernel(kernel_broadcast_rhs, replacements)
    kernel_scalar_rhs = patch_kernel(kernel_scalar_rhs, replacements)

    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs, low=x_low, high=x_high)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if filter_y:
        y[filter_y(y)] = 1
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')

    def do_test(x, y, kernel_fn):
        x_is_scalar = isinstance(x, (bool, int, float))
        y_is_scalar = isinstance(y, (bool, int, float))
        scalar_test = x_is_scalar or y_is_scalar

        # For scalars, we follow the NumPy 2.0 (and JAX/PyTorch pretty much) casting rules.
        if scalar_test:
            # We remove any explicit casting
            pattern = r'\.astype\(np\.\w+\)'
            scalar_expr = expr if numpy_expr is None else re.sub(pattern, '', numpy_expr)
            with promotion_numpy_2_0():
                z_ref = eval(scalar_expr)
        else:
            z_ref = eval(expr if numpy_expr is None else numpy_expr)

        dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
        if not scalar_test and dtype_z is not None:
            z_ref = z_ref.astype(dtype_z)

        # triton result
        x_tri = x if x_is_scalar else to_triton(x, device=device, dst_type=dtype_x)
        y_tri = y if y_is_scalar else to_triton(y, device=device, dst_type=dtype_y)
        z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
        kernel_fn[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
        err_msg = f"{expr}, {kernel_fn.__name__}"
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=err_msg, atol=7e-3, rtol=0.01)

    def get_scalar(x, dtype, low, high, filter):
        # If dtype is int, don't choose a huge number for the scalar
        # as it'll overflow easily when converted to the other dtype
        if dtype in integral_dtypes:
            # Choose in range [-7, 7] ([0, 7] for uints)
            low_x = 0 if dtype in uint_dtypes else -7
            if low is not None:
                low_x = max(low_x, low)
            high_x = 7
            if high is not None:
                high_x = min(high_x, high)
            scalar = numpy_random((), dtype_str=dtype, rs=rs, low=low_x, high=high_x).item()
            if filter and filter(scalar):
                #  https://xkcd.com/221/
                scalar = 4
        else:
            scalar = x.flat[0].item()
        return scalar

    do_test(x, y, kernel)
    if mode_y != 'nan' and test_scalar:
        if dtype_x in uint_dtypes:
            low = 0 if y_low is None else max(y_low, 0)
        else:
            low = y_low
        y_scalar = get_scalar(y, dtype_y, low, y_high, filter_y)
        do_test(x, y_scalar, kernel_scalar_rhs)
    if test_broadcast:
        do_test(x[:1].reshape(()), y, kernel_broadcast_lhs)
        do_test(x, y[:1].reshape(()), kernel_broadcast_rhs)


def _min_max_integral_mod_value(dtype_x, dtype_y) -> tuple[int, int]:
    """
    Limit min/max values for integral types for mod values. Leads to
    overflow/underflow when casting large integral types to floats.
    """
    x_bitwidth = _bitwidth(dtype_x)
    y_bitwidth = _bitwidth(dtype_y)

    # hard cap max value bit-width to 32 if 64 bit-width types
    min_bitwidth = min(x_bitwidth, y_bitwidth, 32)

    # Limit max value bit-width to be one integral type less than the min bit-width
    # For example:
    #   int64, float32 -> int16
    #   uint16, float16 -> uint8
    x_dtype = _dtype(dtype_x)
    max_bitwidth = max(min_bitwidth >> 1, 8)
    dtype_max = x_dtype + str(max_bitwidth)

    max_info = np.iinfo(getattr(np, dtype_max))

    # Still need to limit values here for uints
    if max_bitwidth >= 16 and dtype_max in uint_dtypes:
        return max_info.min, max_info.max // 4
    else:
        return max_info.min, max_info.max


def test_dtype_codegen():
    for dtype in dtypes_with_bfloat16:
        full_name = f"triton.language.{dtype}"
        assert repr(eval(full_name)) == full_name


# ---------------
# test binary ops
# ---------------


def test_unsigned_name_mangling():
    # Test that uint32 and int32 are mangled differently by the compiler
    device = "npu"

    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(to_triton(np.empty_like(e), device=device) for e in expect)
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


# test bitwise ops
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['&', '|', '^']
    for dtype_x in dtypes + dtypes_with_bfloat16
    for dtype_y in dtypes + dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bitwise_op(dtype_x, dtype_y, op, num_ctas):
    device = "npu"

    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if 'float' in dtype_x + dtype_y:
        # The CompilationError must have been caused by a C++ exception with this text.
        with pytest.raises(triton.TritonError, match='invalid operands of type'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr='np.array([])', device=device, num_ctas=num_ctas)
    else:
        _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op) for op in ['<<', '>>'] for dtype_x in int_dtypes + uint_dtypes for dtype_y in uint_dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_shift_op(dtype_x, dtype_y, op, num_ctas):
    device = "npu"

    expr = f'x {op} y'
    bw = max(_bitwidth(dtype_x), _bitwidth(dtype_y))
    if dtype_x.startswith('int'):
        dtype_z = f'int{bw}'
    else:
        dtype_z = f'uint{bw}'
    numpy_expr = f'x.astype(np.{dtype_z}) {op} y.astype(np.{dtype_z})'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas, y_low=0, y_high=bw)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']


# ---------------
# test broadcast
# ---------------


# ----------
# test slice
# ----------


@pytest.mark.interpreter
def test_slice():

    device = "npu"

    @triton.jit
    def slice_kernel(XBLOCK: tl.constexpr):
        data = tl.arange(0, XBLOCK)
        tl.static_assert(data.shape == [XBLOCK])

        t = data[None, :]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, None:]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :None]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None:None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None:None:None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, ::None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None::None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        scalar = tl.full([], 1, tl.int32)
        tl.static_assert(scalar.shape == [])

        t = scalar[None]
        tl.static_assert(t.shape == [1])

        t = scalar[None, None]
        tl.static_assert(t.shape == [1, 1])

    slice_kernel[(1, )](XBLOCK=32)


# ------------------
# test invalid slice
# ------------------


@pytest.mark.interpreter
def test_invalid_slice():
    device = "npu"

    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        dst[10:]

    with pytest.raises(triton.TritonError, match='unsupported tensor index'):
        _kernel[(1, )](dst=dst)


# ----------------
# test expand_dims
# ----------------
@pytest.mark.interpreter
def test_expand_dims():

    device = "npu"

    @triton.jit
    def expand_dims_kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 0)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, 1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, (0, -1))
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.expand_dims(offset1, (0, 1, 3))
        tl.static_assert(t.shape == [1, 1, N, 1])

        t = tl.expand_dims(offset1, (-4, 2, -1))
        tl.static_assert(t.shape == [1, N, 1, 1])

        t = tl.expand_dims(offset1, (3, 1, 2))
        tl.static_assert(t.shape == [N, 1, 1, 1])

        scalar = tl.sum(offset1)
        tl.static_assert(scalar.shape == [])
        t = tl.expand_dims(scalar, 0)
        tl.static_assert(t.shape == [1])

        t = tl.expand_dims(scalar, -1)
        tl.static_assert(t.shape == [1])

        # N is a scalar that's not even a tl.tensor -- this should work too.
        t = tl.expand_dims(N, -1)
        tl.static_assert(t.shape == [1])

    N = 32
    dummy_tensor = torch.empty((), device=device)
    expand_dims_kernel[(1, )](dummy_tensor, N)


@pytest.mark.interpreter
def test_expand_dims_error_cases():

    device = "npu"

    @triton.jit
    def dim_out_of_range1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, -2)
        t = tl.expand_dims(offset1, -3)

    @triton.jit
    def dim_out_of_range2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 1)
        t = tl.expand_dims(offset1, 2)

    @triton.jit
    def dim_out_of_range3(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, 1)
        scalar = tl.sum(offset1)

        t = tl.expand_dims(scalar, 1)

    @triton.jit
    def duplicate_dim1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, 0))

    @triton.jit
    def duplicate_dim2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, -3))

    N = 32
    dummy_tensor = torch.empty((), device=device)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range1[(1, )](dummy_tensor, N)
    assert "invalid axis -3" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range2[(1, )](dummy_tensor, N)
    assert "invalid axis 2" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range3[(1, )](dummy_tensor, N)
    assert "invalid axis 1" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim1[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim2[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))


# ----------------------------
# test invalid program id axis
# ----------------------------
@pytest.mark.interpreter
def test_invalid_pid_axis():
    device = "npu"

    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pid = tl.program_id(20)

    with pytest.raises(triton.TritonError) as exc_info:
        _kernel[(1, )](dst)
    assert re.search(r"program_id axis must be 0, 1, or 2 but got 20", str(exc_info.value.__cause__))


# ---------------
# test where
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where_broadcast(num_ctas):

    device = "npu"

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

        mask = tl.load(cond_ptr + yoffsets)
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    @triton.jit
    def where_scalar_condition(a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = False
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    SIZE = 32
    dtype = 'float32'
    rs = RandomState(17)
    x = numpy_random((SIZE, SIZE), dtype_str=dtype, rs=rs)
    mask = numpy_random(SIZE, 'bool', rs=rs)
    z = np.where(mask, x, 0)
    cond_tri = to_triton(mask, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty((SIZE, SIZE), dtype=z.dtype), device=device, dst_type=dtype)
    where_kernel[(1, )](cond_tri, x_tri, z_tri, SIZE)
    assert (z == to_numpy(z_tri)).all()
    where_scalar_condition[(1, )](x_tri, z_tri, SIZE, num_ctas=num_ctas)
    z = np.where(0, x, 0)
    assert (z == to_numpy(z_tri)).all()


# ---------------
# test unary ops
# ---------------


# ----------------
# test math ops
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("expr", ["tl.math.fdiv(x, y)", "tl.math.div_rn(x, y)"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_math_divide_op(expr, num_ctas):
    device = "npu"

    numpy_expr = "x / y"
    dtype = "float32"
    _test_binary(dtype, dtype, expr, numpy_expr, device=device, num_ctas=num_ctas)


# -------------
# test precise math
# -------------


# ----------------
# test abs
# ----------------


# ----------------
# test passing shapes as individual params rather than tuples
# ----------------


@pytest.mark.interpreter
def test_shapes_as_params():

    device = "npu"

    @triton.jit
    def kernel():
        a = tl.arange(0, 32).expand_dims(-1).broadcast_to(32, 32)
        tl.static_assert(a.shape == [tl.constexpr(32), tl.constexpr(32)])

        a = tl.arange(0, 32).reshape(4, 8).permute(1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).trans()
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).reshape(32)
        tl.static_assert(a.shape == [tl.constexpr(32)])

        a = tl.arange(0, 64).reshape(2, 4, 8).trans(2, 1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4), tl.constexpr(2)])

        a = tl.arange(0, 64).reshape(2, 4, 8).trans((2, 1, 0))
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4), tl.constexpr(2)])

        a = tl.reshape(tl.arange(0, 64), 2, 4, 8, can_reorder=True)
        tl.static_assert(a.shape == [tl.constexpr(2), tl.constexpr(4), tl.constexpr(8)])

    kernel[(1, )]()


# ----------------
# test transpose
# ----------------


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = ttg.convert_layout %3 : tensor<32xi32, #blocked0> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [(f'x[{s}]', d)
                                             for s in ['None, :', ':, None', 'None, :, :', ':, :, None']
                                             for d in ['int32', 'uint32', 'uint16']])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_index1d(expr, dtype_str, num_ctas):
    device = "npu"

    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y - 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x, device=device)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0], num_ctas=num_ctas)
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


@triton.jit(noinline=True)
def noinline_simple_fn(x, y, Z):
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_graph_fn1(x):
    return x + 1


@triton.jit(noinline=True)
def noinline_call_graph_fn2(y):
    return y + 2


@triton.jit(noinline=True)
def noinline_call_graph_fn(x, y, Z):
    t0 = noinline_call_graph_fn1(x)
    t1 = noinline_call_graph_fn2(y)
    z = t0 + t1
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_shared_fn(x, y, Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z) + x + y
    tl.store(Z + offs, z)


@triton.jit(noinline=True)
def noinline_dynamic_fn(x, y, Z):
    if x >= 1:
        x = noinline_call_graph_fn1(x)
    else:
        x = noinline_call_graph_fn2(x)
    if y >= 2:
        y = noinline_call_graph_fn2(y)
    else:
        y = noinline_call_graph_fn1(y)
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_multi_values_fn(x, y):
    return x + 1, y + 2


@triton.jit(noinline=True)
def noinline_multi_values_fn(x, y, Z):
    x, y = noinline_call_multi_values_fn(x, y)
    z = x + y
    tl.store(Z, z)


# ---------------
# test atomics
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_rmw_predicate(num_ctas):

    device = "npu"

    @triton.jit
    def kernel(X):
        val = tl.program_id(0)
        if val < 64:
            tl.atomic_max(X, val)

    x = torch.zeros((1, ), device=device, dtype=torch.int32)
    kernel[(4096, )](x, num_ctas=num_ctas)
    assert x.item() == 63


@pytest.mark.interpreter
@pytest.mark.parametrize("size, num_ctas, dtype_x_str", [(size, num_ctas, dtype_x_str)
                                                         for size in [2, 4, 8, 32, 64, 128]
                                                         for num_ctas in num_ctas_list
                                                         for dtype_x_str in ['bfloat16', 'float16', 'float32']])
def test_tensor_atomic_add_shift_1(size, num_ctas, dtype_x_str):
    device = "npu"

    check_type_supported(dtype_x_str, device)

    @triton.jit
    def kernel(X, val, NUM: tl.constexpr):
        off_x = tl.arange(0, 2)
        off_y = tl.arange(0, NUM)
        off_in = off_x[:, None] * NUM + off_y[None, :]
        off_out = off_x[:, None] + off_y[None, :]

        val = tl.load(val + off_in)
        tl.atomic_add(X + off_out, val)

    s = (2, size)
    dtype = getattr(torch, dtype_x_str)
    x = torch.zeros(s, dtype=dtype, device=device)
    ref = torch.flatten(x)
    val = torch.randn(s, dtype=dtype, device=device)
    kernel[(1, )](x, val, size, num_warps=1, num_ctas=num_ctas)
    val = torch.flatten(val)
    ref[0:size] = val[0:size]
    ref[1:size + 1] += val[size:2 * size]
    torch.testing.assert_close(ref, torch.flatten(x))


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, idx_order, mask_step, num_ctas, dtype_x_str",
                         [(shape, idx_order, mask_step, num_ctas, dtype_x_str)
                          for shape in [(2, 2), (4, 4), (5, 5), (6, 6), (8, 8)]
                          for idx_order in ['increase', 'decrease', 'random_no_duplication', 'random']
                          for mask_step in range(1, 5)
                          for num_ctas in num_ctas_list
                          for dtype_x_str in ['bfloat16', 'float16', 'float32']])
def test_tensor_atomic_add_access_patterns(shape, idx_order, mask_step, num_ctas, dtype_x_str):
    device = "npu"

    check_type_supported(dtype_x_str, device)
    if is_interpreter():
        pytest.skip("not supported in the interpreter")

    @triton.jit
    def kernel(in_ptr, idx_ptr, out_ptr, shape0, shape1, mask_step, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        x_idx = xoffset + tl.arange(0, XBLOCK)[:]
        mask = x_idx < shape0 * shape1
        mask = mask & (x_idx % mask_step != 0)
        idx_base = shape1 * (x_idx // shape1)
        idx_offset = tl.load(idx_ptr + x_idx, mask)
        in_elem = tl.load(in_ptr + x_idx, mask)
        tl.atomic_add(out_ptr + (idx_offset + idx_base), in_elem, mask, sem='relaxed')

    shape0, shape1 = shape
    idx_row = torch.arange(0, shape1, device=device)
    if idx_order == 'increase':
        idx = torch.stack([idx_row.repeat_interleave(i + 1)[:shape1] for i in range(shape0)])
    if idx_order == 'decrease':
        idx = torch.stack([idx_row.flip(0).repeat_interleave(i + 1)[:shape1] for i in range(shape0)])
    if idx_order == 'random_no_duplication':
        idx = torch.stack([torch.randperm(shape1, device=device) for _ in idx_row])
    if idx_order == 'random':
        idx = torch.randint(0, shape1, size=(shape0, shape1), device=device)

    dtype = getattr(torch, dtype_x_str)
    val = torch.randn((shape0, shape1), dtype=dtype, device=device)
    dst = torch.randn((shape0, shape1), dtype=dtype, device=device)

    dst_ref = dst.clone()

    cnt = 0
    for i, row in enumerate(idx):
        for j, elem in enumerate(row):
            if cnt % mask_step != 0:
                dst_ref[i][elem] += val[i][j]
            cnt += 1

    kernel[(1, )](val, idx, dst, shape0, shape1, mask_step, 64, num_ctas=num_ctas)

    if dtype_x_str == 'bfloat16':
        torch.testing.assert_close(dst_ref, dst, rtol=0.1, atol=0.1)
        return

    np.testing.assert_allclose(to_numpy(dst_ref), to_numpy(dst), atol=1e-2)


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_rmw_block(num_ctas):
    device = "npu"

    shape = (8, 8)

    @triton.jit
    def kernel(X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        offs = off0[:, None] * SHAPE1 + off1[None, :]
        val = offs.to(tl.float32)
        x = X + offs
        tl.atomic_min(x, val)

    x = torch.ones((8, 8), device=device, dtype=torch.float32)
    kernel[(2, )](x, shape[0], shape[1], num_ctas=num_ctas)
    assert torch.min(x).item() == 0.0


@pytest.mark.interpreter
@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
                    reason="Requires compute capability >= 9 for NV")
def test_load_scope_sem_coop_grid_cta_not_one():

    device = "npu"

    @triton.jit
    def kernel_r(ptrs, BLOCK_SIZE: tl.constexpr):
        numel = 512
        offset = tl.program_id(0) * BLOCK_SIZE
        index = offset
        mask = index < numel
        a = tl.load(ptrs, mask=mask)
        tl.store(ptrs, a)

    block_size = 128
    data = torch.zeros((128, ), device=device, dtype=torch.float32)

    kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=4, launch_cooperative_grid=True)
    kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=4, launch_cooperative_grid=False)


@pytest.mark.interpreter
def test_load_scope_sem_coop_grid_cta_one():

    device = "npu"

    @triton.jit
    def kernel_r(ptrs, BLOCK_SIZE: tl.constexpr):
        numel = 512
        offset = tl.program_id(0) * BLOCK_SIZE
        index = offset
        mask = index < numel
        a = tl.load(ptrs, mask=mask)
        tl.store(ptrs, a)

    block_size = 128
    data = torch.zeros((128, ), device=device, dtype=torch.float32)

    # Should do nothing different for num_ctas=1 (with coop launch grid)
    kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=1, launch_cooperative_grid=True)
    kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=1, launch_cooperative_grid=False)


@pytest.mark.interpreter
def test_atomic_min_max_neg_zero():

    device = "npu"

    @triton.jit
    def kernel(inp, out_max, out_min):
        idx = tl.program_id(0)
        x = tl.load(inp + idx)
        tl.atomic_max(out_max + idx, x)
        tl.atomic_min(out_min + idx, x)

    N_PROG = 1
    dtype = torch.float32
    out_min = torch.full([N_PROG], torch.finfo(torch.float32).max, device=device, dtype=dtype)
    out_max = torch.full([N_PROG], torch.finfo(torch.float32).min, device=device, dtype=dtype)
    inp = torch.full([N_PROG], -0.0, device=device, dtype=dtype)
    kernel[(N_PROG, )](inp, out_max, out_min)
    torch.testing.assert_close(out_min, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_max, inp, atol=0, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["float8_e4m3fn", "int8", "int16", "uint8", "uint16"])
def test_atomic_unsupported_type(dtype_str):

    device = "npu"

    @triton.jit
    def kernel(I, O):
        x = tl.load(I)
        tl.atomic_add(O, x)

    I = torch.zeros((1, ), device=device, dtype=getattr(torch, dtype_str))
    O = torch.zeros((1, ), device=device, dtype=getattr(torch, dtype_str))
    with pytest.raises(triton.TritonError):
        kernel[(1, )](I, O)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "float16"])
@pytest.mark.parametrize("size", [1, 4, 16])
@pytest.mark.parametrize("op", ["add", "cas"])
def test_tensor_atomic_use_result(dtype_str, size, op):
    device = "npu"

    if is_hip():
        pytest.skip(
            "HIP is broken because (1) it doesn't support thread predicate in atomic cas, and (2) it doesn't support"
            " atomic rmw with float16")

    @triton.jit
    def kernel(index_ptr, out_ptr, size: tl.constexpr, op: tl.constexpr):
        if op == "add":
            write_index = tl.atomic_add(index_ptr + tl.arange(0, size)[:, None], val=tl.arange(0, size)[:, None],
                                        sem="relaxed")
        elif op == "cas":
            write_index = tl.atomic_cas(
                index_ptr + tl.arange(0, size)[:, None],
                cmp=tl.zeros((size, ), dtype=index_ptr.dtype.element_ty)[:, None],
                val=tl.arange(0, size).to(index_ptr.dtype.element_ty)[:, None],
                sem="relaxed",
            )
        tl.store(out_ptr + write_index.to(tl.uint32) * size + tl.arange(0, size)[None, :], 5)

    index = torch.arange(0, size, device=device).to(dtype=getattr(torch, dtype_str))
    out = torch.zeros((size, size), device=device, dtype=getattr(torch, dtype_str))
    kernel[(1, )](index, out, size, op)
    assert (out == 5).all()


# ---------------
# test cast
# ---------------


def test_load_store_same_ptr():

    device = "npu"

    @triton.jit()
    def kernel(in_out_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(in_out_ptr + pid)
        out = x * 2
        tl.store(in_out_ptr + pid, out)

    for _ in range(1000):
        x = torch.ones((65536, ), device=device, dtype=torch.float32)
        if is_hip():
            kernel[(65536, )](x, num_warps=16)  # threads per Warp for ROCM is 64
        else:
            kernel[(65536, )](x, num_warps=32)
        assert torch.all(x == 2)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['int32'])
def test_umulhi(dtype_str):

    device = "npu"

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.umulhi(x, y)
        tl.store(Z + tl.arange(0, N), z)

    def umulhi32(a, b):
        # Convert to 64-bit unsigned integers to prevent overflow
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)

        # Perform the multiplication in 64-bit
        product_64 = a_64 * b_64

        # Shift right by 32 bits to get the high part of the product
        result_high_32 = product_64 >> 32
        return result_high_32

    rs = RandomState(17)
    N = 128
    x = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    x_tri = to_triton(x, device=device)
    y = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    y_tri = to_triton(y, device=device)
    z_tri = torch.zeros_like(x_tri)
    kernel[(1, )](x_tri, y_tri, z_tri, N=N)

    z_ref = umulhi32(x, y)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))


@pytest.mark.interpreter
def test_join():

    device = "npu"

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.join(x, y)
        tl.store(Z + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(-128, 0, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, y, z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_join_scalars():

    device = "npu"

    @triton.jit
    def kernel(X, Y, Z):
        x = tl.load(X)
        y = tl.load(Y)
        z = tl.join(x, y)
        tl.static_assert(z.shape == [2])
        tl.store(Z + tl.arange(0, 2), z)

    x = torch.full([1], 42, device=device).to(torch.int32)
    y = torch.full([1], 100, device=device).to(torch.int32)
    z = torch.zeros([2], device=device)
    kernel[(1, )](x, y, z)

    np.testing.assert_equal([42, 100], to_numpy(z))


@pytest.mark.interpreter
def test_join_with_mma():

    device = "npu"

    @triton.jit
    def kernel(X, Z):
        x = tl.load(X + 16 * tl.arange(0, 32)[:, None] + tl.arange(0, 16)[None, :])  # (32,16)
        x2 = tl.join(x, 2 * x)  # (32,16,2)
        x3 = tl.reshape(x2, (32, 32))
        z = tl.dot(x3, x3)  # (32,32)
        tl.store(Z + 32 * tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :], z)

    x = torch.arange(0, 32 * 16, device=device, dtype=torch.float32).reshape((32, 16))
    r = torch.stack([x, 2 * x], dim=-1).reshape((32, 32))
    z_ref = torch.matmul(r, r)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, z)

    torch.testing.assert_close(z, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("debug", [False, True])
def test_interleave(debug):

    device = "npu"

    @triton.jit(debug=debug)
    def kernel(Z, N: tl.constexpr):
        z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(128, 256, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1).reshape(256)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_interleave_scalars():

    device = "npu"

    @triton.jit
    def kernel(X, Y, Z):
        z = tl.interleave(X, Y)
        tl.static_assert(z.shape == [tl.constexpr(2)])
        tl.store(Z + tl.arange(0, 2), z)

    z = torch.zeros(2, device=device)
    kernel[(1, )](10, 20, z)

    np.testing.assert_equal([10, 20], to_numpy(z))


@pytest.mark.interpreter
def test_split():

    device = "npu"

    @triton.jit
    def kernel(X, Z1, Z2, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        x1 = tl.reshape(x, (N // 2, 2))
        z1, z2 = tl.split(x1)
        tl.store(Z1 + tl.arange(0, N // 2), z1)
        tl.store(Z2 + tl.arange(0, N // 2), z2)

    x = torch.arange(0, 256, device=device).to(torch.int32).reshape((128, 2))
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2, N=256)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


@pytest.mark.interpreter
def test_split_to_scalar():

    device = "npu"

    @triton.jit
    def kernel(X, Z1, Z2):
        offs = tl.arange(0, 2)
        x = tl.load(X + offs)
        z1, z2 = tl.split(x)
        tl.static_assert(isinstance(z1, tl.tensor))
        tl.static_assert(isinstance(z2, tl.tensor))
        tl.static_assert(z1.shape == [])
        tl.static_assert(z2.shape == [])
        tl.store(Z1, z1)
        tl.store(Z2, z2)

    N = 2
    x = torch.arange(0, N, device=device).reshape(N // 2, 2)
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(
        exp == 0,
        # subnormal
        ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (frac / (2.0**dtype.fp_mantissa_width)),
        # normal
        ((-1.0)**sign) * (2.0**(exp - exp_bias)) * (1.0 + frac / (2.0**dtype.fp_mantissa_width))).float()

    extended_exp = (
        (1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4nv, tl.float8e4b15]:
        # float8e4m3nv does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp
                              | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width)))  #
                             .view(torch.float32), output)
    return output


@pytest.mark.interpreter
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_convert_float16_to_float32(in_dtype):
    """Tests that check convert_float_to_float32 function"""
    device = "npu"

    check_type_supported(in_dtype, device)

    f16_input = torch.tensor(range(-int(2**(16 - 1)), int(2**(16 - 1))), dtype=torch.int16).view(in_dtype)
    f32_output = convert_float_to_float32(f16_input)

    nan = f16_input.isnan()
    assert torch.all(f32_output[nan].isnan())
    inf = f16_input.isinf()
    assert torch.all(f32_output[inf].isinf())
    other = torch.logical_not(torch.logical_or(nan, inf))
    assert torch.all(f16_input[other] == f32_output[other])


# ---------------
# test reduce
# ---------------


@pytest.mark.interpreter
def test_max_returns_zero():
    # Simple test with a tl.max call that returns 0.  The interpreter had a bug
    # where it didn't handle this correctly.
    device = "npu"

    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        z = tl.max(x)
        tl.store(Z, z)

    BLOCK = 128
    x = torch.zeros((BLOCK, ), device=device)
    z = torch.ones((1, ), device=device)

    kernel[(1, )](x, z, BLOCK=BLOCK)
    assert z[0] == 0


@pytest.mark.interpreter
def test_max_min_with_nan():
    # In triton, we implement a "nan ignore" style, which means if there is NaN
    # in the reduce dimesion, we should ignore it and return the max/min number,
    # it's different with torch.max/min.
    device = "npu"

    @triton.jit
    def max_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        max_val = tl.max(x, axis=0)

        if tl.program_id(0) == 0:
            tl.store(y_ptr, max_val)

    @triton.jit
    def min_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        min_val = tl.min(x, axis=0)

        if tl.program_id(0) == 0:
            tl.store(y_ptr, min_val)

    BLOCK_SIZE = 64
    x = torch.rand((1, BLOCK_SIZE), dtype=torch.float32, device=device)
    # Not the expected output for tl.max
    x[0, 0] = float('nan')
    # Expected output for tl.min
    x[0, 1] = float('-inf')
    # Expected output for tl.max
    x[0, 2] = float('inf')

    y = torch.ones(1, device=device)

    max_kernel[(1, )](x, y, BLOCK_SIZE=BLOCK_SIZE)
    assert y[0] == float('inf')

    min_kernel[(1, )](x, y, BLOCK_SIZE=BLOCK_SIZE)
    assert y[0] == float('-inf')


def get_reduced_dtype(dtype_str, op):
    if op in ('argmin', 'argmax'):
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


def get_reduce_input(dtype_str, shape):
    # limit the range of integers so that reduce ops do not overflow
    low = 0 if dtype_str in uint_dtypes else -10 if dtype_str in integral_dtypes else None
    high = 10 if dtype_str in integral_dtypes else None
    return numpy_random(shape, dtype_str=dtype_str, low=low, high=high)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [(op, dtype, (1, 1024), axis, False)
                   for dtype in dtypes_with_bfloat16
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for axis in [1]]

# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
# , (32, 64), (64, 128)]
if is_cuda() and 'V100' in torch.cuda.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]

reduce_configs2 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce2d_shapes
                   for axis in [0, 1]] + [(op, 'float32', [16, 32], None, False) for op in ['min', 'max', 'sum']]

reduce3d_shapes = [(2, 32, 16), (32, 2, 16), (32, 16, 2)]
reduce_configs3 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce3d_shapes
                   for axis in [0, 1, 2]]
invalid_config = [('sum', 'float32', (32, 32), axis, False) for axis in [2, 3]]
negative_config = [('sum', 'float32', (32, 32), -1, False)]
keep_dims_2d_configs = [(op, 'float32', (32, 32), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1]] + [(op, 'float32', (32, 32), None, True) for op in ['min', 'max', 'sum']]
keep_dims_3d_configs = [(op, 'float32', (32, 2, 16), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1, 2]] + [(op, 'float32', (32, 2, 16), None, True)
                                                  for op in ['min', 'max', 'sum']]
reduce_bool = [(op, 'bool', shape, axis, False) for op in ['xor_sum'] for shape in reduce2d_shapes for axis in [0, 1]]


scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [(op, type, shape, axis, reverse, num_warps)
                for num_warps in [4, 16]
                for type in ['int32', 'float32', 'bfloat16']
                for axis in [1, 0]
                for reverse in [True, False]
                for shape in scan2d_shapes
                for op in ['cumsum', 'cumprod', 'get_first_element', 'linear_recurrence', 'cummax', 'roll']]
negative_config = [('cumsum', 'float32', (32, 32), -1, False, 4)]


# trivial associative but not commutative function
@triton.jit
def get_first_element(a, b):
    return a


# Compute x_i = a_i * x_{i-1} + b_i
@triton.jit
def linear_recurrence(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2


@triton.jit
def cummax(v0, i0, v1, i1):
    gt = v0 > v1
    return tl.where(gt, v0, v1), tl.where(gt, i0, i1)


@triton.jit
def roll(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


# ---------------
# test histogram
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[2048, 2], [1024, 8], [1024, 128], [256, 512], [32, 512], [8, 512], [8, 2]])
def test_histogram(M, N):

    device = "npu"

    @triton.jit
    def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + offset1)
        z = tl.histogram(x, N)
        bias = tl.full([M, N], 1, dtype=tl.int32)
        # check that histogram produces object compatible with broadcasting
        biased = z + bias
        tl.store(z_ptr + offset2, z)

    torch.manual_seed(17)
    x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
    z = torch.empty(N, dtype=torch.int32, device=device)
    # torch.histc does not work when the input type is not float and the device is CPU
    # https://github.com/pytorch/pytorch/issues/74236
    # This is a workload by converting the input to float
    z_torch = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    histogram_kernel[(1, )](x, z, M=M, N=N)
    assert (z_torch == z).all()


@pytest.mark.interpreter
def test_histogram_silent_data_corruption():

    device = "npu"

    @triton.jit
    def histogram_kernel(x_ptr, z_ptr):
        offset = tl.arange(0, 1)
        x = tl.load(x_ptr + offset)
        z = tl.histogram(x, 1)
        tl.store(z_ptr + offset, z)

    x = torch.ones(1, device=device, dtype=torch.int32)
    z = torch.ones(2, device=device, dtype=torch.int32)

    histogram_kernel[(1, )](x, z)
    assert z[1] == 1, f"Second element shouldn't be affected, expected_buffer=[1, 1], actual_buffer={z}"


# ------------------------
# test histogram with mask
# ------------------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[2048, 2], [1024, 8], [1024, 128], [256, 512], [32, 512], [8, 512], [8, 2]])
def test_histogram_mask(M, N):

    device = "npu"

    @triton.jit
    def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, 2 * M)
        offset2 = tl.arange(0, N)
        mask = offset1 < M
        x = tl.load(x_ptr + offset1)
        z = tl.histogram(x, N, mask)
        tl.store(z_ptr + offset2, z)

    torch.manual_seed(17)
    x1 = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
    x = torch.cat((x1, x1), 0)
    z = torch.empty(N, dtype=torch.int32, device=device)
    # torch.histc does not work when the input type is not float and the device is CPU
    # https://github.com/pytorch/pytorch/issues/74236
    # This is a workload by converting the input to float
    z_torch = torch.histc(x1.float(), bins=N, min=0, max=N - 1)
    histogram_kernel[(1, )](x, z, M=M, N=N)
    assert (z_torch == z).all()


@pytest.mark.parametrize("M, N", [(1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)])
def test_scan_1d(M, N):

    device = "npu"

    @triton.jit
    def scan_kernel(out_ptr, in_ptr, M: tl.constexpr, N: tl.constexpr):
        input = tl.load(in_ptr + tl.arange(0, M))
        output = tl.cumsum(input).reshape([1, M]).broadcast_to([N, M])
        tl.store(out_ptr + tl.arange(0, M * N), output.reshape([M * N]))

    x = torch.randint(-100, 100, (M, ), dtype=torch.int32, device=device)
    output = torch.empty(M * N, dtype=torch.int32, device=device)

    scan_kernel[(1, )](output, x, M, N)

    ref = torch.cumsum(x, dim=0).reshape([1, M]).broadcast_to([N, M]).reshape([M * N])
    torch.testing.assert_close(ref.to(torch.int32), output, atol=0, rtol=0)


@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def _sum_combine(a, b):
    return a + b


@pytest.mark.interpreter
def test_generic_reduction():

    device = "npu"

    @triton.jit
    def var_mean_kernel(X, out_mean, out_var, out_sum0, out_sum1, BLOCK: tl.constexpr):
        xindex = tl.arange(0, BLOCK)
        x = tl.load(X + xindex)
        mean = x
        m2 = tl.zeros_like(x)
        weight = tl.full(x.shape, 1, x.dtype)
        # Test return a tuple and a single value
        sum0, = tl.reduce((x, ), 0, _sum_combine)
        sum1 = tl.reduce(x, 0, _sum_combine)
        # Test multiple values in a tuple
        (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine)
        tl.store(out_mean, mean)
        tl.store(out_var, m2 / weight)
        tl.store(out_sum0, sum0)
        tl.store(out_sum1, sum1)

    SIZE = 512
    x = torch.rand(SIZE, device=device)
    out_mean = torch.empty((), device=device)
    out_var = torch.empty((), device=device)
    sum0 = torch.empty((), device=device)
    sum1 = torch.empty((), device=device)

    var_mean_kernel[(1, )](x, out_mean, out_var, sum0, sum1, BLOCK=SIZE)

    expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
    sum_ref = torch.sum(x)
    torch.testing.assert_close(out_mean, expect_mean)
    torch.testing.assert_close(out_var, expect_var)
    torch.testing.assert_close(sum0, sum_ref)
    torch.testing.assert_close(sum1, sum_ref)


# ---------------
# test permute
# ---------------


# ---------------
# test dot
# ---------------


def convert_fp8_to_fp32(x, device, dtype_str):
    if dtype_str == 'float8e4nv':
        return torch.tensor(x, device=device).view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype_str == 'float8e5':
        return torch.tensor(x, device=device).view(torch.float8_e5m2).to(torch.float32)
    elif dtype_str == 'float8e4b8':
        return torch.tensor(x, device=device).view(torch.float8_e4m3fnuz).to(torch.float32)
    elif dtype_str == 'float8e5b16':
        return torch.tensor(x, device=device).view(torch.float8_e5m2fnuz).to(torch.float32)
    raise AssertionError("Unsupported float8 dtype")


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_base_cases():
    return [(*shape, 4, False, False, epilogue, input_precision, in_dtype, out_dtype, 1, None)
            for shape in [(64, 64, 64), (32, 32, 32), (16, 16, 16)]
            for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax', 'chain-dot']
            for input_precision in ['tf32', 'tf32x3', 'ieee', 'bf16x3', 'bf16x6']
            for in_dtype, out_dtype in [('float16', 'float16'), ('float16', 'float32'), ('float32', 'float32')]
            if not (input_precision != 'ieee' and (in_dtype in ['float16']))]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_softmax():
    return [(128, 128, 64, 8, False, False, 'softmax', 'ieee', 'float16', 'float32', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_mixed_sizes_cases():
    available_kpack = [1, 2 if (is_hip() and not is_hip_cdna4()) else 1]
    available_precision = ["tf32" if is_cuda() else "ieee"]
    return [
        (*shape_nw, col_a, col_b, 'none', input_precision, in_dtype, out_dtype, kpack, None)
        for shape_nw in [[128, 256, 32, 8], [128, 16, 32, 4], [32, 128, 64, 4], [128, 128, 64, 4], [64, 128, 128, 4],
                         [32, 128, 64, 2], [64, 64, 32, 4], [32, 32, 128, 16], [128, 128, 64, 2], [64, 128, 128, 2]]
        for input_precision in available_precision
        for col_a in [True, False]
        for col_b in [True, False]
        for in_dtype, out_dtype in [('int8', 'int8'), ('float16', 'float16'), ('float16',
                                                                               'float32'), ('float32', 'float32')]
        for kpack in available_kpack
    ]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #2370
def get_test_dot_transposed_op_base_cases():
    return [(64, 64, 64, 4, col_a, col_b, 'none', 'ieee', 'float32', 'float32', 1, None)
            for col_a in [True, False]
            for col_b in [True, False]]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# Introduced in #2750
def get_test_dot_h100_shortcut_cases():
    return [(64, 64, 64, 4, False, False, 'chain-dot', 'ieee', 'bfloat16', 'float32', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #3908
def get_test_dot_mfma_edge_cases():
    if not is_hip_cdna():
        return []
    return [(16, 16, 8, 4, False, False, 'None', 'ieee', 'float32', 'float32', 1, None),
            (32, 16, 8, 4, False, False, 'None', 'ieee', 'float16', 'float16', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #3370
def get_test_dot_fp8_output_cases():
    return [(128, 128, 64, 4, False, False, 'chain-dot', 'ieee', float8_type, 'float32', 1, None)
            for float8_type in ["float8e5", "float8e4nv"]]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #5406
def get_test_dot_small_k_mfma_cases():
    if not is_hip_cdna():
        return []
    return [(32, 32, k_size, 4, False, False, 'None', 'ieee', in_dtype, out_dtype, 1, mma_nonk_size)
            for k_size in [1, 2, 4, 8]
            for in_dtype, out_dtype in [('float16', 'float32'), ('int8', 'int32')]
            for mma_nonk_size in mma_nonk_sizes]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #4516
def get_test_dot_small_mn_mfma_cases():
    if not is_hip_cdna():
        return []
    return [(*shape_nw, False, False, epilogue, 'ieee', in_dtype, out_dtype, 1, None)
            for shape_nw in [(4, 64, 64, 1), (64, 4, 64, 1)]
            for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols']
            for in_dtype, out_dtype in [('float16', 'float16'), ('float32', 'float32')]]


def get_test_dot_double_rate_cases():
    if not is_hip_cdna():
        return []
    return [(32, 32, 16, 4, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (32, 32, 16, 4, False, False, 'None', 'ieee', 'bfloat16', 'float32', 1, None),
            (16, 16, 32, 4, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (16, 16, 32, 4, False, False, 'None', 'ieee', 'bfloat16', 'float32', 1, None)]


def get_test_dot_vdot2_cases():
    if not is_hip_cdna():
        return []
    return [(4, 32, 32, 4, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (4, 32, 32, 4, False, False, 'None', 'ieee', 'bfloat16', 'float32', 1, None)]


def get_test_small_dots_cases():
    if not is_cuda():
        return []
    return [(2, 4, 32, 1, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (1, 2, 32, 1, False, False, 'None', 'ieee', 'float8e5', 'float32', 1, None)]


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str",
    [(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str)
     for B in [1, 2, 4, 8]
     for num_warps in [1, 2, 4, 8, 16]
     for BLOCK_M, BLOCK_N in [(32, 32)]
     for M, N, K in [(64, 64, 64), (32, 32, 32)]
     for in_dtype_str, out_dtype_str in [('int8', 'int8'), ('float16', 'float16'), ('float16', 'float32'),
                                         ('float32', 'float32')]] +
    # Large block sizes
    [(4, 4, 128, 128, 64, 64, 64, 'float16', 'float16')] +
    # Small block sizes
    [(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str)
     for B in [1, 2, 8]
     for num_warps in [1, 2, 4]
     for BLOCK_M, BLOCK_N in [(1, 32), (32, 2), (8, 8)]
     for M, N, K in [(32, 32, 32)]
     for in_dtype_str, out_dtype_str in [('float16', 'float16'), ('float32', 'float32')]])
def test_dot3d(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str):
    device = "npu"

    if is_hip():
        # hip does not support tf32 precision, so use ieee for all tests
        input_precision = "ieee"
        arch = triton.runtime.driver.active.get_current_target().arch
        if "gfx11" in arch or "gfx12" in arch:
            if in_dtype_str == "float32":
                pytest.skip(f"{in_dtype_str} is not supported in WMMA dot, FMA does not support dot3d")
            if out_dtype_str == "float16":
                pytest.skip(f"{out_dtype_str} has low precision in WMMA dot")
        if in_dtype_str == "float64":
            pytest.skip("float64 not supported on HIP yet")
    else:
        input_precision = "tf32" if is_cuda() and in_dtype_str == 'float32' else "ieee"
        if not is_interpreter() and (BLOCK_M < 16 or BLOCK_N < 16):
            pytest.skip("small dots are supported only on HIP at the moment")

    shared_mem_accum = B * (BLOCK_M * K + K * BLOCK_N) * get_src_element_ty_size(in_dtype_str)
    if not is_interpreter() and triton.runtime.driver.active.utils.get_device_properties(
            triton.runtime.driver.active.get_current_device())["max_shared_mem"] < shared_mem_accum:
        pytest.skip("Skipped due to insufficient shared memory on this GPU.")

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        o_ptr,
        stride_qb,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kk,
        stride_kn,
        stride_ob,
        stride_om,
        stride_on,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        out_dtype: tl.constexpr = tl.float32,
    ):
        startm = tl.program_id(0) * BLOCK_M
        startn = tl.program_id(1) * BLOCK_N
        offs_b = tl.arange(0, BLOCK_B)
        offs_m = startm + tl.arange(0, BLOCK_M)
        offs_n = startn + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        q_ptrs = q_ptr + offs_b[:, None, None] * stride_qb + offs_m[None, :, None] * stride_qm + offs_k[
            None, None, :] * stride_qk
        k_ptrs = k_ptr + offs_b[:, None, None] * stride_kb + offs_k[None, :, None] * stride_kk + offs_n[
            None, None, :] * stride_kn
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        o_ptrs = o_ptr + offs_b[:, None, None] * stride_ob + offs_m[None, :, None] * stride_om + offs_n[
            None, None, :] * stride_on
        tl.store(o_ptrs, qk)

    if out_dtype_str == 'int8':
        out_dtype = tl.int8
    elif out_dtype_str == 'float16':
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    rs = RandomState(17)
    x = numpy_random((B, M, K), dtype_str=in_dtype_str, rs=rs)
    y = numpy_random((B, K, N), dtype_str=in_dtype_str, rs=rs)
    if in_dtype_str == 'int8':
        out = numpy_random((B, M, N), dtype_str='int32', rs=rs)
    else:
        if is_hip() and (BLOCK_M < 16 or BLOCK_N < 16) and out_dtype_str == 'float16':
            # float16 accumulator in FMA dot loose precision too fast
            x *= 0.1
            y *= 0.1
        out = numpy_random((B, M, N), dtype_str=out_dtype_str, rs=rs)

    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    out_tri = to_triton(out, device=device)

    BLOCK_B = B
    BLOCK_K = K

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    kernel[grid](
        x_tri,
        y_tri,
        out_tri,
        x_tri.stride(0),
        x_tri.stride(1),
        x_tri.stride(2),
        y_tri.stride(0),
        y_tri.stride(1),
        y_tri.stride(2),
        out_tri.stride(0),
        out_tri.stride(1),
        out_tri.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        INPUT_PRECISION=input_precision,
        out_dtype=out_dtype,
        num_warps=num_warps,
    )

    if in_dtype_str == 'int8':
        out_ref = np.matmul(x.astype(np.float32), y.astype(np.float32)).astype(np.int32)
    else:
        out_ref = np.matmul(x, y)
    np.testing.assert_allclose(out_ref, to_numpy(out_tri), rtol=0.01, atol=1e-2)


@pytest.mark.parametrize('in_dtype', ['float32'])
def test_dot_mulbroadcasted(in_dtype):
    device = "npu"

    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            pytest.skip("Requires sm >= 80 to run")

    @triton.jit
    def kernel(Z, X, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
        pidn = tl.program_id(1)
        pidm = tl.program_id(0)
        offm = tl.arange(0, BM)[:, None]
        offn = tl.arange(0, BN)[None, :]
        offak = tl.arange(0, BK)[None, :]
        offbk = tl.arange(0, BK)[:, None]
        acc = tl.full((BM, BN), 0.0, tl.float32)
        for ridx5 in range(0, K // BK):
            x = tl.load(X + ((pidm * K * BM) + (offm * K) + (ridx5 * BK) + offak))
            y = tl.load(Y + ((pidn * BN) + (offbk * N) + (ridx5 * N * BK) + offn))
            x = tl.expand_dims(x, axis=2)
            y = tl.expand_dims(y, axis=0)
            t = tl.sum(x * y, axis=1)
            acc = t + acc
        tl.store(Z + ((pidm * BM * N) + (pidn * BN) + (offm * N) + offn), acc)

    M, N, K = 256, 192, 160
    BM, BN, BK = 128, 32, 32
    rs = RandomState(17)
    x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    x = x * 0.1
    y = y * 0.1
    z = numpy_random((M, N), dtype_str=in_dtype, rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(z, device=device)
    grid = M // BM, N // BN
    h = kernel[grid](z_tri, x_tri, y_tri, M, N, K, BM, BN, BK)
    z_ref = np.matmul(x, y)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), atol=0.01)

    if not is_cuda():
        return
    assert "tt.dot" in h.asm['ttir']
    assert re.search(r"ttg.async_wait %.* {num = 2 : i32}", h.asm["ttgir"]) is not None


@triton.jit
def pass_const(a, b, choose_b):
    if choose_b:
        return b
    else:
        return a


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
def test_dot_without_load(dtype_str):

    device = "npu"

    @triton.jit
    def _kernel(out):
        a = GENERATE_TEST_HERE
        b = GENERATE_TEST_HERE
        c = tl.dot(a, b)
        out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(out_ptr, c)

    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
    a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    out_ref = torch.matmul(a, b)
    out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](out)
    assert torch.all(out == out_ref)


# ---------------
# test arange
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("start", [0, 1, 7, 16])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_arange(start, num_ctas):
    device = "npu"

    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)

    _kernel[(1, )](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK, num_ctas=num_ctas)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    np.testing.assert_allclose(to_numpy(z_tri), to_numpy(z_ref))


# ---------------
# test load
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
@pytest.mark.parametrize("mask_val", [True, False])
@pytest.mark.parametrize("other_val", [0, 1])
def test_masked_load_scalar(num_ctas, mask_val, other_val):
    device = "npu"

    input_val = 4.0
    size = 128
    dtype = torch.float32
    input = torch.full((size, ), input_val, dtype=dtype, device=device)
    output = torch.zeros((size, ), dtype=dtype, device=device)

    @triton.jit
    def kernel(in_ptr, out_ptr, size: tl.constexpr, mask: tl.constexpr, other: tl.constexpr):
        offsets = tl.arange(0, size)
        x = tl.load(in_ptr + offsets, mask=mask, other=other)
        tl.store(out_ptr + offsets, x)

    kernel[(1, )](input, output, size, mask_val, other_val, num_ctas=num_ctas)

    if mask_val:
        reference_out = torch.full((size, ), input_val, dtype=dtype, device=device)
    else:
        reference_out = torch.full((size, ), other_val, dtype=dtype, device=device)

    torch.testing.assert_close(output, reference_out)


# Testing masked loads with a copy to shared memory.
# FIXME: Shape too small for ldmatrix when num_ctas=4
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype):

    device = "npu"

    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    M = 32
    N = 32
    K = 16

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr, in_stride, in2_stride, out_stride, in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
        in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < M * K)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < K * N)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w, out_dtype=tl.float32)

        # Store output
        output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < M * N)

    pgm = _kernel[(1, )](in1, in2, out, in1.stride()[0], in2.stride()[0], out.stride()[0], in1.numel(), in2.numel(),
                         out.numel(), M=M, N=N, K=K)

    reference_out = torch.matmul(in1, in2)
    torch.testing.assert_close(out, reference_out, atol=1e-2, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("cache", ["", ".ca", ".cg", ".cv"])
def test_load_cache_modifier(cache):
    device = "npu"

    src = torch.empty(128, device=device)
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets, cache_modifier=CACHE)
        tl.store(dst + offsets, x)

    pgm = _kernel[(1, )](dst, src, CACHE=cache)

    if is_hip():
        target_arch = get_arch()
        # TODO: support testing for remaining architectures
        if 'gfx94' not in target_arch:
            return
        amdgcn = pgm.asm['amdgcn']
        cg_cache_modifier_str = 'nt'
        cv_cache_modifier_str = 'sc0 sc1'
        buffer_load_line = [line for line in amdgcn.splitlines() if "buffer_load" in line]
        global_load_line = [line for line in amdgcn.splitlines() if "global_load" in line]
        load_line = global_load_line[0] if global_load_line else buffer_load_line[0]
        if cache == '' or cache == '.ca':
            assert cg_cache_modifier_str not in load_line
        if cache == '.cg':
            assert cg_cache_modifier_str in load_line
        if cache == '.cv':
            assert cv_cache_modifier_str in load_line

    if is_cuda():
        ptx = pgm.asm['ptx']
        if cache == '':
            assert 'ld.global.ca' not in ptx
            assert 'ld.global.cg' not in ptx
        if cache == '.cg':
            assert 'ld.global.cg' in ptx
            assert 'ld.global.ca' not in ptx
        if cache == '.ca':
            assert 'ld.global.ca' in ptx
            assert 'ld.global.cg' not in ptx


@pytest.mark.interpreter
@pytest.mark.parametrize("N", [16, 10, 11, 1024])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_vectorization(N, num_ctas):
    device = "npu"

    block_size = 1024 * num_ctas
    src = torch.randn(block_size, device=device)
    dst = torch.empty(block_size, device=device)

    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    pgm = _kernel[(1, )](dst, src, N=N, BLOCK_SIZE=block_size)

    if not is_cuda():
        return

    ptx = pgm.asm["ptx"]
    if N % 16 == 0:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.b32" in ptx
    torch.testing.assert_close(dst[:N], src[:N], atol=1e-6, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("has_hints", [False, True])
def test_vectorization_hints(has_hints):
    device = "npu"

    src = torch.empty(1024, device=device)
    dst = torch.empty(1024, device=device)
    off = torch.zeros(1, device=device, dtype=torch.int32)

    @triton.jit
    def _kernel(dst, src, off, N, BLOCK_SIZE: tl.constexpr, HINT: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offsets = offsets + tl.load(off)
        if HINT:
            tl.max_contiguous(tl.multiple_of(offsets, 1024), 1024)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    pgm = _kernel[(1, )](dst, src, off, N=1024, BLOCK_SIZE=src.shape[0], HINT=has_hints)
    if not is_cuda():
        return

    ptx = pgm.asm["ptx"]
    if has_hints:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.v4.b32" not in ptx


# ---------------
# test store
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("cache", ["", ".wb", ".cg", ".cs", ".wt"])
def test_store_cache_modifier(cache):
    device = "npu"

    src = torch.empty(128, device=device)
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets)
        tl.store(dst + offsets, x, cache_modifier=CACHE)

    pgm = _kernel[(1, )](dst, src, CACHE=cache)

    if is_hip():
        target_arch = get_arch()
        # TODO: support testing for remaining architectures
        if 'gfx94' not in target_arch:
            return
        amdgcn = pgm.asm['amdgcn']
        cs_cache_modifier_str = 'nt'
        wt_cache_modifier_str = 'sc0 sc1'
        buffer_store_line = [line for line in amdgcn.splitlines() if "buffer_store" in line]
        global_store_line = [line for line in amdgcn.splitlines() if "global_store" in line]
        store_line = global_store_line[0] if global_store_line else buffer_store_line[0]
        if cache == '' or cache == '.cg':
            assert cs_cache_modifier_str not in store_line
            assert wt_cache_modifier_str not in store_line
        if cache == '.cs':
            assert cs_cache_modifier_str in store_line
            assert wt_cache_modifier_str not in store_line
        if cache == '.wt':
            assert cs_cache_modifier_str not in store_line
            assert wt_cache_modifier_str in store_line

    if is_cuda():
        ptx = pgm.asm['ptx']
        if cache == '':
            assert 'st.global.wb' not in ptx
            assert 'st.global.cg' not in ptx
            assert 'st.global.cs' not in ptx
            assert 'st.global.wt' not in ptx
        if cache == '.wb':
            assert 'st.global.wb' in ptx
            assert 'st.global.cg' not in ptx
            assert 'st.global.cs' not in ptx
            assert 'st.global.wt' not in ptx
        if cache == '.cg':
            assert 'st.global.wb' not in ptx
            assert 'st.global.cg' in ptx
            assert 'st.global.cs' not in ptx
            assert 'st.global.wt' not in ptx
        if cache == '.cs':
            assert 'st.global.wb' not in ptx
            assert 'st.global.cg' not in ptx
            assert 'st.global.cs' in ptx
            assert 'st.global.wt' not in ptx
        if cache == '.wt':
            assert 'st.global.wb' not in ptx
            assert 'st.global.cg' not in ptx
            assert 'st.global.cs' not in ptx
            assert 'st.global.wt' in ptx


@pytest.mark.interpreter
@pytest.mark.parametrize("eviction_policy", ["", "evict_last", "evict_first"])
def test_store_eviction_policy(eviction_policy):
    device = "npu"

    src = torch.empty(128, device=device)
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst, src, POLICY: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets)
        tl.store(dst + offsets, x, eviction_policy=POLICY)

    pgm = _kernel[(1, )](dst, src, POLICY=eviction_policy)

    if not is_cuda():
        return
    ptx = pgm.asm['ptx']
    if eviction_policy == '':
        assert 'evict_last' not in ptx
        assert 'evict_first' not in ptx
    if eviction_policy == 'evict_last':
        assert 'evict_last' in ptx
        assert 'evict_first' not in ptx
    if eviction_policy == 'evict_first':
        assert 'evict_last' not in ptx
        assert 'evict_first' in ptx


# ---------------
# test default
# ---------------
# TODO: can't be local to test_default


@triton.jit
def _impl(value=10):
    return value


@pytest.mark.interpreter
def test_default():
    device = "npu"

    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device=device)
    ret1 = torch.zeros(1, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(ret0, ret1, value=3):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1, )](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

    _kernel[(1, )](ret0, ret1)
    assert ret0.item() == 10
    assert ret1.item() == 3


# ---------------
# test noop
# ----------------


# --------------------
# value specialization
# --------------------


@pytest.mark.parametrize("value, value_type", [(-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
                                               (2**31, 'i64'), (2**32 - 1, 'i64'), (2**32, 'i64'), (2**63 - 1, 'i64'),
                                               (-2**63, 'i64'), (2**63, 'u64'), (2**64 - 1, 'u64')])
def test_value_specialization(value: int, value_type: str) -> None:

    device = "npu"

    def repr(specialization):
        ty = specialization.signature["value1"]
        cst = '_'.join([k for k, v in specialization.constants.items() if isinstance(k, str) and v == 1])
        return f"kernel_{ty}_{cst}"

    @triton.jit(repr=repr)
    def kernel(value1, is_one, X):
        pass

    x = torch.tensor([3.14159], device=device)
    h = kernel.warmup(value, 1, x, grid=(1, ))
    assert "is_one" in h.name
    assert value_type in h.name


@pytest.mark.parametrize("value, overflow", [(2**64 - 1, False), (2**64, True), (-2**63, False), (-2**63 - 1, True)])
def test_value_specialization_overflow(value: int, overflow: bool) -> None:

    device = "npu"

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device=device)

    if overflow:
        with pytest.raises(OverflowError):
            kernel[(1, )](value, x)
    else:
        kernel[(1, )](value, x)


# ----------------
# test constexpr
# ----------------


@pytest.mark.interpreter
def test_constexpr_shape():

    device = "npu"

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1, )](x_tri)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256))


@pytest.mark.interpreter
def test_constexpr_scalar_shape():

    device = "npu"

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1, )](x_tri, 32)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256) % 8)


reshape_list = [((64, ), (8, 8)), ((2, 32), (16, 4)), ((512, ), (2, 2, 2, 2, 2, 2, 2, 2, 2)), ((64, 32), (16, 8, 16))]


@pytest.mark.interpreter
@pytest.mark.parametrize("formats", reshape_list)
def test_reshape(formats):
    device = "npu"

    in_format, out_format = formats

    @triton.jit
    def kernel(Z, X, out_tuple: tl.constexpr):
        x = tl.load(X_PTR_EXPR)
        z = tl.reshape(x, out_tuple)
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
        }
        return patch_kernel(kernel, to_replace)

    x = numpy_random(in_format, dtype_str="int32")
    z = x.reshape(out_format)
    x_tri = to_triton(x, device=device)
    patched_kernel = generate_kernel(in_format, out_format)
    z_tri = to_triton(np.empty(out_format, dtype=np.int32), device=device)
    patched_kernel[(1, )](z_tri, x_tri, out_format)
    np.testing.assert_equal(z, to_numpy(z_tri))


def test_reshape_err():

    device = "npu"

    @triton.jit
    def kernel():
        x = tl.arange(0, 8 * 8)
        y = tl.reshape(x, (8 * 4, ))

    with pytest.raises(triton.CompilationError) as exc_info:
        kernel.warmup(grid=(1, ))

    assert "reshape" in str(exc_info.value)


@pytest.mark.interpreter
def test_tma_load_block_shape_err():

    device = "npu"

    @triton.jit
    def kernel(ptr):
        desc = tl.make_tensor_descriptor(ptr, [128, 128], [128, 1], [1, 2])
        desc.load([0, 0])

    input = torch.empty((128, 128), dtype=torch.int32, device=device)
    errc = triton.CompilationError if not is_interpreter() else InterpreterError
    with pytest.raises(errc) as e:
        kernel[(1, )](input)

    assert "Descriptor block shape must have at least 16 bytes" in str(e.value.__cause__)


@pytest.mark.interpreter
def test_tma_store_block_shape_err():

    device = "npu"

    @triton.jit
    def kernel(ptr):
        desc = tl.make_tensor_descriptor(ptr, [128, 128], [128, 1], [8, 4])
        desc.store([0, 0], tl.zeros([8, 4], dtype=tl.int16))

    input = torch.empty((128, 128), dtype=torch.int16, device=device)
    errc = triton.CompilationError if not is_interpreter() else InterpreterError
    with pytest.raises(errc) as e:
        kernel[(1, )](input)

    assert "Descriptor block shape must have at least 16 bytes" in str(e.value.__cause__)


# -------------
# test call
# -------------


@triton.jit
def val_multiplier(val, i):
    return val * i


@triton.jit(noinline=True)
def val_multiplier_noinline(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep, type: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        if type == "inline":
            vec = val_multiplier(vec, i)
        else:
            vec = val_multiplier_noinline(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


@pytest.mark.interpreter
@pytest.mark.parametrize("type", ["inline", "noinline"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_call(type, num_ctas):

    device = "npu"

    @triton.jit
    def kernel(ptr, n_elements, num1, num2, type: tl.constexpr):
        vecmul_kernel(ptr, n_elements, num1, type)
        vecmul_kernel(ptr, n_elements, num2, type)

    size = 1024
    rand_val = numpy_random((size, ), dtype_str="float32")
    rand_val_tri = to_triton(rand_val, device=device)
    err_msg = ""
    try:
        kernel[(size // 128, )](rand_val_tri, size, 3, 5, type, num_ctas=num_ctas)
    except Exception as e:
        err_msg = str(e)

    if type == "noinline" and not is_interpreter():
        assert err_msg != ""
    else:
        ans = rand_val * 1 * 2 * 1 * 2 * 3 * 4
        np.testing.assert_equal(to_numpy(rand_val_tri), ans)


# -------------
# test if
# -------------


@pytest.mark.interpreter
@pytest.mark.parametrize("if_type", [
    "if", "if_and_dynamic", "if_exp_static", "if_exp_dynamic", "if_exp_dynamic_constexpr", "if_exp_dynamic_void",
    "if_and_static"
])
def test_if(if_type):

    device = "npu"

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret, IfType: tl.constexpr, BoolVar: tl.constexpr, StaticValue: tl.constexpr):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if IfType == "if":
            if pid % 2 == 0:  # eq
                tl.store(Ret, tl.load(XTrue))
            elif 1 == pid % 2:  # req
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_dynamic":
            val = tl.load(XTrue) if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_constexpr":
            val = 3.14 if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_void":
            tl.store(Ret, tl.load(XTrue)) if pid % 2 == 0 else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_static":
            tl.store(Ret, tl.load(XTrue)) if BoolVar else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_dynamic":
            if BoolVar and (1 != pid % 2 and pid % 2 != 1):  # rne and ne
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_static":
            if StaticValue != 0 and StaticValue != 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device=device)
    x_true = torch.tensor([3.14], dtype=torch.float32, device=device)
    x_false = torch.tensor([1.51], dtype=torch.float32, device=device)
    ret = torch.zeros(1, dtype=torch.float32, device=device)

    kernel[(1, )](cond, x_true, x_false, ret, if_type, True, 1)
    assert torch.equal(ret, x_true)


# -----------------------
# test inline asm
# -----------------------


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_inline_asm(num_ctas):
    device = "npu"

    if not is_cuda():
        pytest.skip("test_inline_asm is only supported in NPU")

    @triton.jit
    def kernel(X, Y, Z, n: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        s = tl.full([BLOCK], n, tl.int32)
        z = tl.inline_asm_elementwise("shf.l.wrap.b32 $0, $1, $2, $3;", "=r,r, r, r", [x, y, s], dtype=tl.int32,
                                      is_pure=True, pack=1)
        tl.store(Z + tl.arange(0, BLOCK), z)

    shape = (128, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='uint32', rs=rs)
    y = numpy_random(shape, dtype_str='uint32', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    n = 17
    z_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, n, BLOCK=shape[0], num_ctas=num_ctas)
    y_ref = (y << n) | (x >> (32 - n))
    # compare
    np.testing.assert_equal(y_ref, to_numpy(z_tri))


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_inline_asm_packed(num_ctas):
    device = "npu"

    if not is_cuda():
        pytest.skip("test_inline_asm is only supported in NPU")

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        # shift 4x8bits values together.
        y = tl.inline_asm_elementwise(
            "and.b32 $0, $1, 0x1F1F1F1F; \
                                       shl.b32 $0, $0, 3;", "=r,r", [
                x,
            ], dtype=tl.int8, is_pure=True, pack=4)
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (512, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='uint8', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(numpy_random(shape, dtype_str='uint8', rs=rs), device=device)
    kernel[(1, )](x_tri, y_tri, BLOCK=shape[0], num_ctas=num_ctas)
    y_ref = x << 3
    # compare
    np.testing.assert_equal(y_ref, to_numpy(y_tri))


@pytest.mark.parametrize('num_ctas', num_ctas_list)
def test_inline_asm_with_pointers(num_ctas):
    device = "npu"

    if not is_cuda():
        pytest.skip('test_inline_asm is only supported in NPU')

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x_ptrs = X + tl.arange(0, BLOCK)
        y_ptrs = Y + tl.arange(0, BLOCK)
        tl.inline_asm_elementwise(
            "ld.global.b8 $0, [$1]; \
                                   shl.b32 $0, $0, 3; \
                                   st.global.b8 [$2], $0;", "=r,l,l", [x_ptrs, y_ptrs], dtype=tl.int8, is_pure=False,
            pack=1)

    shape = (512, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='uint8', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(numpy_random(shape, dtype_str='uint8', rs=rs), device=device)
    kernel[(1, )](x_tri, y_tri, BLOCK=shape[0], num_ctas=num_ctas)
    y_ref = x << 3
    # compare
    np.testing.assert_equal(y_ref, to_numpy(y_tri))


def test_inline_asm_multiple_outputs():
    device = "npu"

    if not is_cuda():
        pytest.skip('test_inline_asm is only supported in NPU')

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))

        # C = A - B
        # D = B - A
        (c, d) = tl.inline_asm_elementwise(
            asm="""
            sub.u32 $0, $2, $3;  // C = A - B
            sub.u32 $1, $3, $2;  // D = B - A
            """,
            constraints=(
                # 2 output registers: $0=C and $1=D.
                "=r,=r,"
                # 2 input registers: $2=A and $3=B.
                "r,r"),
            args=[a, b],
            dtype=(tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )
        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    rs = RandomState(17)
    A = numpy_random(shape, dtype_str='uint32', rs=rs)
    B = numpy_random(shape, dtype_str='uint32', rs=rs)
    A_tri = to_triton(A, device=device)
    B_tri = to_triton(B, device=device)
    C_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    D_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    kernel[(1, )](A_tri, B_tri, C_tri, D_tri, BLOCK=shape[0])

    C_ref = A - B
    D_ref = B - A

    np.testing.assert_equal(C_ref, to_numpy(C_tri))
    np.testing.assert_equal(D_ref, to_numpy(D_tri))


def test_inline_asm_packed_multiple_outputs():
    device = "npu"

    if not is_cuda():
        pytest.skip('test_inline_asm is only supported in NPU')

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))

        # For each (a,b) in zip(a,b), perform the following:
        # - Let ai be `a` converted to int32.
        # - Let af be `a` converted to float.
        # - Let m be the max of ai and b.
        # - Return ai and mi.
        # Do the above 4 elements at a time.
        (c, d) = tl.inline_asm_elementwise(
            asm="""
            {
                // Unpack `a` into `ai`.
                .reg .b8 tmp<4>;
                mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                cvt.u32.u8 $0, tmp0;
                cvt.u32.u8 $1, tmp1;
                cvt.u32.u8 $2, tmp2;
                cvt.u32.u8 $3, tmp3;
            }
            // Convert `ai` to float.
            cvt.rn.f32.s32 $4, $0;
            cvt.rn.f32.s32 $5, $1;
            cvt.rn.f32.s32 $6, $2;
            cvt.rn.f32.s32 $7, $3;
            // Take max of `ai` and `b`.
            max.f32 $4, $4, $9;
            max.f32 $5, $5, $10;
            max.f32 $6, $6, $11;
            max.f32 $7, $7, $12;
            """,
            constraints=(
                # 8 output registers, namely
                #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
                #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
                "=r,=r,=r,=r,=r,=r,=r,=r,"
                # 5 input registers, namely
                #   $8=ai,
                #   $9=b0, $10=b1, $11=b2, $12=b3.
                # The four elements from `a` are all packed into one register.
                "r,r,r,r,r"),
            args=[a, b],
            dtype=(tl.int32, tl.float32),
            is_pure=True,
            pack=4,
        )
        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    rs = RandomState(17)
    A = numpy_random(shape, dtype_str='uint8', rs=rs)
    B = numpy_random(shape, dtype_str='float32', rs=rs)
    A_tri = to_triton(A, device=device)
    B_tri = to_triton(B, device=device)
    C_tri = to_triton(numpy_random(shape, dtype_str='int32', rs=rs), device=device)
    D_tri = to_triton(numpy_random(shape, dtype_str='float32', rs=rs), device=device)
    kernel[(1, )](A_tri, B_tri, C_tri, D_tri, BLOCK=shape[0])

    C_ref = A.astype(np.int32)
    D_ref = np.maximum(A.astype(np.float32), B)

    np.testing.assert_equal(C_ref, to_numpy(C_tri))
    np.testing.assert_equal(D_ref, to_numpy(D_tri))


# -----------------------
# test map elementwise
# -----------------------


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_map_elementwise(num_ctas):

    device = "npu"

    @triton.jit
    def compare(x, y):
        if x < y:
            return -1
        elif x == y:
            return 0
        else:
            return 1

    @triton.jit
    def kernel(X, Y, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        z = tl.map_elementwise(compare, x, y)
        tl.store(Z + tl.arange(0, BLOCK), z)

    shape = (128, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='int32', rs=rs)
    y = numpy_random(shape, dtype_str='int32', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(numpy_random(shape, dtype_str='int32', rs=rs), device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, BLOCK=shape[0], num_ctas=num_ctas)
    z_ref = (x > y).astype(int) - (y > x).astype(int)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))


def test_map_elementwise_multiple_outputs():

    device = "npu"

    @triton.jit
    def divmod(a, b):
        return a // b, a % b

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))

        c, d = tl.map_elementwise(divmod, a, b)

        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    rs = RandomState(17)
    A = numpy_random(shape, dtype_str='uint32', rs=rs)
    B = numpy_random(shape, dtype_str='uint32', rs=rs)
    A_tri = to_triton(A, device=device)
    B_tri = to_triton(B, device=device)
    C_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    D_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    kernel[(1, )](A_tri, B_tri, C_tri, D_tri, BLOCK=shape[0])

    C_ref = A // B
    D_ref = A % B

    np.testing.assert_equal(C_ref, to_numpy(C_tri))
    np.testing.assert_equal(D_ref, to_numpy(D_tri))


def test_map_elementwise_pack():

    device = "npu"

    @triton.jit
    def divmod(a0, a1, b0, b1):
        return a0 // b0, a1 // b1, a0 % b0, a1 % b1

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))

        c, d = tl.map_elementwise(divmod, a, b, pack=2)

        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    rs = RandomState(17)
    A = numpy_random(shape, dtype_str='uint32', rs=rs)
    B = numpy_random(shape, dtype_str='uint32', rs=rs)
    A_tri = to_triton(A, device=device)
    B_tri = to_triton(B, device=device)
    C_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    D_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    h = kernel[(1, )](A_tri, B_tri, C_tri, D_tri, BLOCK=shape[0])

    C_ref = A // B
    D_ref = A % B

    np.testing.assert_equal(C_ref, to_numpy(C_tri))
    np.testing.assert_equal(D_ref, to_numpy(D_tri))


# -----------------------
# test control flow
# -----------------------


@pytest.mark.interpreter
def test_if_else():

    device = "npu"

    @triton.jit
    def kernel(Cond, TrueVal, FalseVal, Out):
        if tl.load(Cond):
            val = tl.load(TrueVal)
        else:
            val = tl.load(FalseVal)
        tl.store(Out, val)

    out = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    true_val = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    false_val = to_triton(np.full((1, ), 2, dtype=np.int32), device=device)
    cond = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    # True
    cond[0] = True
    kernel[(1, )](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == true_val[0]
    # False
    cond[0] = False
    kernel[(1, )](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == false_val[0]


@pytest.mark.interpreter
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_if_return(mode):

    device = "npu"

    @triton.jit
    def kernel(ExitEarly, Out, cond: tl.constexpr, mode: tl.constexpr):
        if mode == "dynamic":
            if tl.load(ExitEarly):
                tl.store(Out, 0)
                return
        else:
            if cond:
                tl.store(Out, 0)
                return
        tl.store(Out, 1)

    out = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    exit_early = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    # exit early path taken
    exit_early[0] = 1
    kernel[(1, )](exit_early, out, True, mode)
    assert to_numpy(out)[0] == 0
    # exit early path not taken
    exit_early[0] = 0
    kernel[(1, )](exit_early, out, False, mode)
    assert to_numpy(out)[0] == 1


@triton.jit
def add_fn(x):
    return x + 1


@triton.jit(noinline=True)
def add_fn_noinline(x):
    return x + 1


@triton.jit
def add_fn_return(x, pid):
    if pid == 0:
        return x + 1
    else:
        return x + 2


@triton.jit
def add_fn_expr(Out, x):
    tl.store(Out, x)


@triton.jit
def add_fn_static_cond(x, cond: tl.constexpr):
    if cond == "":
        return x
    else:
        return x + 1


@pytest.mark.interpreter
@pytest.mark.parametrize("_cond1", [True, False])
@pytest.mark.parametrize("_cond2", [True, False])
@pytest.mark.parametrize("_cond3", [True, False])
def test_nested_if_else_return(_cond1, _cond2, _cond3):

    device = "npu"

    @triton.jit
    def kernel(Cond1, Cond2, Cond3, Val1, Val2, Val3, Out):
        val = 0
        if tl.load(Cond1):
            if tl.load(Cond2):
                val = tl.load(Val1)
            else:
                return
        else:
            if tl.load(Cond3):
                val = tl.load(Val2)
            else:
                val = tl.load(Val3)
        tl.store(Out, val)

    out = to_triton(np.full((1, ), -1, dtype=np.int32), device=device)
    cond1 = to_triton(np.full((1, ), _cond1, dtype=np.int32), device=device)
    cond2 = to_triton(np.full((1, ), _cond2, dtype=np.int32), device=device)
    cond3 = to_triton(np.full((1, ), _cond3, dtype=np.int32), device=device)
    val1 = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    val2 = to_triton(np.full((1, ), 2, dtype=np.int32), device=device)
    val3 = to_triton(np.full((1, ), 3, dtype=np.int32), device=device)
    kernel[(1, )](cond1, cond2, cond3, val1, val2, val3, out)
    targets = {
        (True, True, True): val1[0],
        (True, True, False): val1[0],
        (True, False, True): out[0],
        (True, False, False): out[0],
        (False, True, True): val2[0],
        (False, True, False): val3[0],
        (False, False, True): val2[0],
        (False, False, False): val3[0],
    }
    assert out[0] == targets[(_cond1, _cond2, _cond3)]


@pytest.mark.interpreter
def test_while():

    device = "npu"

    @triton.jit
    def kernel(InitI, Bound, CutOff, OutI, OutInitI, OutJ):
        init_i = tl.load(InitI)
        curr_i = init_i
        j = 0
        # Check that init_i is not updated by the loop
        while j < tl.load(Bound):
            curr_i = curr_i + (j == tl.load(CutOff))
            j += 1
            tl.store(OutInitI, init_i)
        tl.store(OutI, curr_i)
        tl.store(OutJ, j)

    out_i = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    out_j = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    init_i = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    out_init_i = to_triton(np.full((1, ), 0, dtype=np.int32), device=device)
    bound = to_triton(np.full((1, ), 10, dtype=np.int32), device=device)
    cut_off = to_triton(np.full((1, ), 5, dtype=np.int32), device=device)
    kernel[(1, )](init_i, bound, cut_off, out_i, out_init_i, out_j)
    assert out_init_i[0] == init_i[0]
    assert out_i[0] == init_i[0] + 1
    assert out_j[0] == bound[0]


@pytest.mark.interpreter
def test_nested_while():

    device = "npu"

    @triton.jit
    def nested_while(data, countPtr):
        for i in range(10):
            count = tl.load(countPtr)
            while count > 0:
                tl.store(data, tl.load(data) + 1.0)
                count = count - 2

    counter = torch.tensor([8], dtype=torch.int32, device=device)
    data = torch.zeros((1, ), device=device, dtype=torch.float32)
    nested_while[(1, )](data, counter)
    assert data[0] == 40


def test_constexpr_if_return():
    # Reproducer for #4883, return statement in an if with a constexpr causes
    # errors when combined with non-trivial control flow graphs

    device = "npu"

    @triton.jit
    def kernel(Semaphore, Out, total: tl.constexpr):
        if total == 1:
            tl.store(Out, tl.program_id(0))
            return

        prev = tl.atomic_add(Semaphore, 1)
        if prev + 1 != total:
            return

        tl.store(Out, tl.program_id(0) + prev)

    sem = torch.zeros((), device=device, dtype=torch.int32)
    out = torch.empty((), device=device, dtype=torch.int32)
    kernel[(1, )](sem, out, 1)
    assert out.item() == 0

    sem = torch.zeros((), device=device, dtype=torch.int32)
    out = torch.full((), fill_value=-1, device=device, dtype=torch.int32)
    kernel[(4, )](sem, out, 4)
    assert out.item() >= 0


def test_constexpr_flattens():
    assert tl.constexpr(tl.constexpr(5)) == tl.constexpr(5)
    assert tl.constexpr(tl.constexpr(tl.constexpr(5))) == tl.constexpr(5)


@pytest.mark.parametrize("literal, tensor_ty", [(10, tl.int32), (32.1, tl.float32),
                                                ((5, 6, 7), None),  # tuples can't be lifted to tensors
                                                ])
def test_constexpr_assignment(literal, tensor_ty):
    from triton.language.core import constexpr_type

    @triton.jit
    def kernel(input_literal: tl.constexpr, tensor_type: tl.constexpr):
        patched_literal: tl.constexpr = PATCHED
        # Sanity checks
        tl.static_assert(patched_literal.type == constexpr_type(PATCHED))
        tl.static_assert(input_literal.type == constexpr_type(PATCHED))

        assigned_literal: tl.constexpr = input_literal
        tl.static_assert(assigned_literal.type == constexpr_type(PATCHED))
        tl.static_assert(assigned_literal == patched_literal)

        if tensor_type is not None:
            assigned_variable = input_literal
            tl.static_assert(assigned_variable.type == tensor_type)

    kernel_patched = patch_kernel(kernel, {'PATCHED': f"{literal}"})
    kernel_patched[(1, )](literal, tensor_ty)


@triton.jit
def return_poison(x):
    a = False
    if a:
        return x


# -----------------------
# test extra
# -----------------------


def test_num_threads():
    device = "npu"

    if is_hip():
        pytest.skip("test_num_threads is not supported in HIP")

    @triton.jit
    def kernel(Out):
        num_threads: tl.constexpr = tl.extra.cuda.num_threads()
        offs = tl.arange(0, num_threads)
        tl.store(Out + offs, 1)

    num_threads = 256
    out = to_triton(np.zeros((num_threads, ), dtype=np.int32), device=device)
    kernel[(1, )](out, num_warps=num_threads // 32)
    assert torch.sum(out) == 256


def test_globaltimer():
    device = "npu"

    check_npu_or_hip(device)
    if is_hip():
        pytest.skip("test_globaltimer is flaky on AMD GPUs")

    @triton.jit
    def kernel(Out1, Out2, func: tl.constexpr):
        start = func()
        off = tl.arange(0, 128)
        for i in range(10000):
            tl.store(Out1 + off, tl.load(Out1 + off) + 1)
        end = func()
        tl.store(Out2, start)
        tl.store(Out2 + 1, end)

    out1 = to_triton(np.zeros((128, ), dtype=np.int64), device=device)
    out2 = to_triton(np.zeros((2, ), dtype=np.int64), device=device)
    if is_cuda():
        func = tl.extra.cuda.globaltimer
    else:
        func = tl.extra.hip.memrealtime
    h = kernel[(1, )](out1, out2, func)
    assert out2[1] - out2[0] > 0
    if is_cuda():
        assert h.asm["ptx"].count("%globaltimer") == 2
    else:
        target_arch = triton.runtime.driver.active.get_current_target().arch
        if "gfx11" in target_arch or "gfx12" in target_arch:
            assert h.asm["amdgcn"].count("s_sendmsg_rtn_b64") == 2
        else:
            assert h.asm["amdgcn"].count("s_memrealtime") == 2


def test_smid():
    device = "npu"

    if is_hip():
        pytest.skip("test_smid is not supported in HIP")
    check_npu_or_hip(device)

    @triton.jit
    def kernel(Out):
        tl.store(Out + tl.program_id(0), tl.extra.cuda.smid())

    out = to_triton(np.zeros((1024, ), dtype=np.int32), device=device)
    h = kernel[(out.shape[0], )](out)
    assert out.sort()[0].unique().shape[0] > 0
    assert h.asm["ptx"].count("%smid") == 1


@pytest.mark.interpreter
def test_load_scalar_with_mask():

    device = "npu"

    @triton.jit
    def kernel(Input, Index, Out, N: int):
        index = tl.load(Index)
        scalar = tl.load(Input + index, mask=index < N, other=0)
        tl.store(Out, scalar, mask=index < N)

    Index = torch.tensor([0], dtype=torch.int32, device=device)
    Input = torch.tensor([0], dtype=torch.int32, device=device)
    Out = torch.empty_like(Index, device=device)
    kernel[(1, )](Input, Index, Out, Index.numel())
    assert Out.data[0] == 0


# This test is used to test our own PTX codegen for float16 and int16 conversions
# maybe delete it later after ptxas has been fixed


# -----------------------
# test fp8 -> fp32 dot
# -----------------------


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        low_precision_acc: tl.constexpr,  #
        num_stages: tl.constexpr = 3  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=num_stages):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, max_num_imprecise_acc=low_precision_acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


# -----------------------
# test enable_fp_fusion
# -----------------------


@pytest.mark.parametrize("enable_fp_fusion", [False, True])
@pytest.mark.parametrize("default_override", [False, True])
def test_enable_fp_fusion(enable_fp_fusion, default_override, fresh_knobs):
    # Sequential multiply add can be fused by backend
    device = "npu"

    @triton.jit
    def mul_add(data):
        ptrs = data + tl.arange(0, 128)
        tl.store(ptrs, tl.load(ptrs) * 1.5 + 1.0)

    data = torch.randn((128, ), device=device, dtype=torch.float32)
    if default_override:
        fresh_knobs.language.default_fp_fusion = enable_fp_fusion
        h = mul_add.warmup(data, grid=(1, ))
    else:
        h = mul_add.warmup(data, grid=(1, ), enable_fp_fusion=enable_fp_fusion)

    if not is_cuda():
        return
    found_fma = re.search(r'(mad|fma)\.r[nzmp]\.(ftz\.)?f32', h.asm["ptx"]) is not None
    assert found_fma == enable_fp_fusion


# -----------------------
# test enable_reflect_ftz
# -----------------------


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("enable_reflect_ftz", [False, True])
def test_enable_reflect_ftz(enable_reflect_ftz, fresh_knobs):

    device = "npu"

    @triton.jit
    def exp2(data):
        ptrs = data + tl.arange(0, 128)
        tl.store(ptrs, tl.math.exp2(tl.load(ptrs)))

    data = torch.full((128, ), -127.0, device=device, dtype=torch.float32)
    h = exp2.warmup(data, grid=(1, ), enable_reflect_ftz=enable_reflect_ftz)

    found_ex2_ftz = re.search(r'ex2.approx.ftz.f32', h.asm["ptx"]) is not None
    assert found_ex2_ftz == enable_reflect_ftz


# -----------------------
# test override_arch
# -----------------------


@pytest.mark.parametrize("arch", ["sm70", "sm80", "sm90", "gfx942", "gfx950", "gfx1200"])
@pytest.mark.parametrize("env_var_override", [False, True])
def test_override_arch(arch, env_var_override, fresh_knobs):
    device = "npu"

    if arch.startswith("sm") and not is_cuda():
        pytest.skip(f"{arch} arch only for CUDA")
    elif arch.startswith("gfx") and not is_hip():
        pytest.skip(f"{arch} arch only for HIP")

    @triton.jit
    def simple(data, out):
        in_ptrs = data + tl.arange(0, 128)
        out_ptrs = out + tl.arange(0, 128)
        tl.store(out_ptrs, tl.load(in_ptrs) * 1.5 + 1.0)

    data = torch.randn((128, ), device=device, dtype=torch.float32)
    out = torch.empty_like(data)

    if is_cuda():
        if env_var_override:
            fresh_knobs.runtime.override_arch = str(arch)
            h = simple.warmup(data, out, grid=(1, ))
        else:
            h = simple.warmup(data, out, arch=arch, grid=(1, ))
        ttgir_cc = re.search(r'cuda:(\d+)', h.asm["ttgir"])
        assert ttgir_cc.group(1) == arch[2:]
    elif is_hip():
        # For HIP, the generated kernel is a binary containing the final ISA. So we cannot run
        # them like CUDA side if the chip doesn't match. Here we just check generated ISA.
        if env_var_override:
            fresh_knobs.runtime.override_arch = str(arch)
            h = simple.warmup(data, out, grid=(1, ))
        else:
            h = simple.warmup(data, out, arch=arch, grid=(1, ))
        ttgir_gfx = re.search(r'hip:(\w+)', h.asm["ttgir"])
        ttgir_warp = re.search(r'"ttg.threads-per-warp" = (\d+)', h.asm["ttgir"])
        amdgcn_gfx = re.search(r'.amdgcn_target "amdgcn-amd-amdhsa--(\w+)"', h.asm["amdgcn"])
        assert ttgir_gfx.group(1) == arch
        assert int(ttgir_warp.group(1)) == (32 if arch == "gfx1200" else 64)
        assert amdgcn_gfx.group(1) == arch


def test_num_ctas_pre_sm90():
    device = "npu"

    if not is_cuda() and not is_hip():
        pytest.skip("Only supported on CUDA and HIP")

    @triton.jit
    def _kernel(src):
        pass

    src = torch.empty(1, device=device)
    if is_cuda():
        arch = "sm80"
        msg = r"num_ctas > 1 requires NVIDIA SM90\+ \(Hopper\)"
    else:
        arch = "gfx942"
        msg = r"num_ctas > 1 not supported"

    with pytest.raises(ValueError, match=msg):
        _kernel.warmup(src, grid=(1, ), num_ctas=2, arch=arch)


# -----------------------
# test propagate_nan
# -----------------------


# -----------------------
# test clamp
# -----------------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ['float16', 'float32'])
def test_clamp(dtype):

    device = "npu"

    @triton.jit
    def kernel(x_ptr, min_ptr, max_ptr, out_ptr, ref_ptr, N, BLOCK_SIZE: tl.constexpr):
        off = tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        _min = tl.load(min_ptr + off, mask=mask)
        _max = tl.load(max_ptr + off, mask=mask)
        out = out_ptr + off
        ref = ref_ptr + off

        tl.store(out, tl.clamp(x, _min, _max), mask=mask)
        ref_val = tl.minimum(tl.maximum(x, _min), _max)
        tl.store(ref, ref_val, mask=mask)

    size = 128

    x = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    a = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    b = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    _min = torch.min(a, b)
    _max = torch.max(a, b)
    out = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))
    ref = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))

    kernel[(size, )](x, _min, _max, out, ref, x.numel(), BLOCK_SIZE=size)

    torch.testing.assert_close(out, ref)


# Test for symmetric clamp(x, -limit, limit), as it may go through optimized
# codegen in the backends
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ['bfloat16', 'float16', 'float32'])
def test_clamp_symmetric(dtype):

    device = "npu"

    @triton.jit
    def kernel(x_ptr, limit_ptr, out_ptr, ref_ptr, N, BLOCK_SIZE: tl.constexpr):

        off = tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        limit = tl.load(limit_ptr + off, mask=mask)
        out = out_ptr + off
        ref = ref_ptr + off

        tl.store(out, tl.clamp(x, -limit, limit), mask=mask)
        ref_val = tl.minimum(tl.maximum(x, -limit), limit)
        tl.store(ref, ref_val, mask=mask)

    size = 128

    x = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    limit = torch.randn((size, ), device=device, dtype=getattr(torch, dtype)).abs()
    out = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))
    ref = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))

    kernel[(size, )](x, limit, out, ref, x.numel(), BLOCK_SIZE=size)

    torch.testing.assert_close(out, ref)


# -----------------------
# test iterators
# -----------------------


@pytest.mark.interpreter
def test_static_range():

    device = "npu"

    @triton.jit
    def loop_kernel(Z, N: tl.constexpr, step: tl.constexpr):
        acc = 0
        for i in tl.static_range(0, N, step=step):
            acc += i
        tl.store(Z, acc)

    N = 100
    step = 7
    Out = torch.empty(1, dtype=torch.int32, device=device)
    loop_kernel[(1, )](Out, N, step)
    Acc = torch.tensor([0], dtype=torch.int32, device=device)
    for i in range(0, N, step):
        Acc += i
    assert (Out == Acc).all(), (Out, Acc)


@pytest.mark.interpreter
def test_tl_range_num_stages():
    device = "npu"

    if is_hip():
        pytest.skip("test_tl_range is not supported in HIP")
    M, N, K = 64, 64, 512
    BLOCK_M, BLOCK_N, BLOCK_K = M, N, 64
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    pgm = matmul_kernel[
        1,
    ](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M, BLOCK_N,
      BLOCK_K, 0, num_stages=5)
    ref_out = torch.matmul(a, b).to(torch.float32)
    if is_interpreter():
        # GPU invokes tensor core for float16 matmul, which is not supported in interpreter.
        # Thus we use a higher tolerance
        torch.testing.assert_close(ref_out, c, rtol=1e-2, atol=1e-1)
    else:
        torch.testing.assert_close(ref_out, c, rtol=1e-3, atol=1e-3)
        if device in ['cuda']:
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                ptx = pgm.asm['ptx']
                # check that the loop got pipelined with the right number of stages.
                assert 'cp.async.wait_group \t6' in ptx


def test_tl_range_option_none():

    @triton.jit
    def kernel(ub):
        for i in tl.range(0, ub, num_stages=None, loop_unroll_factor=None):
            print("i", i)

    compiled_kernel = kernel.warmup(10, grid=(1, ))
    assert "num_stages" not in compiled_kernel.asm["ttir"]
    assert "loop_unroll_factor" not in compiled_kernel.asm["ttir"]


@triton.jit(noinline=True)
def maxnreg_noinline1(X):
    tl.store(X, 0)


@triton.jit(noinline=True)
def maxnreg_noinline2(X):
    tl.store(X, 0)


@pytest.mark.interpreter
def test_maxnreg():
    device = "npu"

    if not is_cuda():
        pytest.skip('maxnreg only works on CUDA')

    # triton kernel
    @triton.jit
    def kernel(X):
        maxnreg_noinline1(X)
        tl.store(X, 0)
        maxnreg_noinline2(X)

    X = torch.empty(1, dtype=torch.int32, device=device)
    k = kernel[(1, )](X, maxnreg=42)

    if not is_interpreter():
        # Ensure that .maxnreg is set on the kernel function (marked with .entry)
        # and not on either of the noinline functions (marked with .func).
        try:
            assert re.search(r'\.visible \.entry [^{;]*\.maxnreg 42', k.asm["ptx"])
            assert not re.search(r'\.visible \.func [^{;]*\.maxnreg', k.asm["ptx"])
        except AssertionError:
            print("Failing ptx:\n", k.asm["ptx"])
            raise


@pytest.mark.interpreter
def test_temp_var_in_loop():

    device = "npu"

    @triton.jit
    def temp_in_loop(Z, N: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.full((BLOCK, ), 0, dtype=tl.int32)
        for i in range(N):
            if i == 0:
                temp = tl.full((BLOCK, ), 2, dtype=tl.int32)
                acc = temp
            else:
                acc += tl.full((BLOCK, ), 1, dtype=tl.int32)
            # reuse the temp variable and make sure to check that it isn't creating incorrect IR.
            temp = tl.full((BLOCK, ), 1, dtype=tl.int32)
            acc += temp
        z = Z + tl.arange(0, BLOCK)
        tl.store(z, acc)

    N = 10
    BLOCK = 32
    out = torch.empty((BLOCK, ), dtype=torch.int32, device=device)
    temp_in_loop[(1, )](out, N, BLOCK)
    acc = torch.full((BLOCK, ), 0, dtype=torch.int32, device=device)
    for i in range(N):
        if i == 0:
            temp = torch.full((BLOCK, ), 2, dtype=torch.int32, device=device)
            acc = temp
        else:
            acc += torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        temp = torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        acc += temp
    assert (acc == out).all()


@pytest.mark.interpreter
def test_num_programs():
    # Assuming that the kernel is launched with a grid of (11, 21, 31)
    device = "npu"

    grid = (11, 21, 31)
    input = torch.empty((3, ), dtype=torch.int32, device=device)

    @triton.jit
    def kernel(input):
        num_programs_0 = tl.num_programs(0)
        num_programs_1 = tl.num_programs(1)
        num_programs_2 = tl.num_programs(2)
        tl.store(input, num_programs_0)
        tl.store(input + 1, num_programs_1)
        tl.store(input + 2, num_programs_2)

    kernel[grid](input)
    assert torch.all(input == torch.tensor(grid, device=device))


# -----------------------
# test loop unrolling
# -----------------------


def test_unroll_attr():

    device = "npu"

    @triton.jit
    def _kernel(dst, unroll_factor: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in tl.range(0, 10, loop_unroll_factor=unroll_factor):
            tl.atomic_add(dst + pid, i + pid)

    def check_loop_unroll_count(ir, opStr, loop_unroll_factor):
        for line in ir.splitlines():
            if opStr in line:
                loop_unroll_factor = loop_unroll_factor - 1
        # Sometimes we get a remainder loop
        assert loop_unroll_factor <= 0

    # Try for all different loop unroll factors (compile-only):
    tmp = torch.empty(1, device=device)
    for unroll_factor in [1, 2, 4, 5, 8]:
        h = _kernel.warmup(tmp, unroll_factor, grid=(1, ))
        check_loop_unroll_count(h.asm["ttir"], 'tt.atomic_rmw', unroll_factor)


@triton.jit
def sanitize_add(a, b):
    a64 = a.to(tl.int64)
    b64 = b.to(tl.int64)
    r64 = a64 + b64
    tl.device_assert((r64 >= -2**31) & (r64 <= 2**31 - 1))
    return a + b


def test_side_effectful_reduction():
    device = "npu"

    if device != "npu":
        pytest.skip()

    @triton.jit(debug=True)
    def sanitize_sum_kernel(Z, X, BLOCK: tl.constexpr):
        vals = tl.load(X + tl.arange(0, BLOCK))
        z = tl.reduce(vals, 0, sanitize_add)
        tl.store(Z, z)

    BLOCK = 512
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK], device="npu", dtype=torch.int32)
    X[:300] = 32
    X[300:] = 0
    Z = torch.zeros((), device="npu", dtype=torch.int32)
    sanitize_sum_kernel[(1, )](Z, X, BLOCK=BLOCK)
    torch.testing.assert_close(Z, X.sum().to(torch.int32))


@pytest.mark.parametrize("reduce_dim", [0, 1])
def test_side_effectful_reduction_2d(reduce_dim):
    device = "npu"

    if device != "npu":
        pytest.skip()

    @triton.jit(debug=True)
    def sanitize_sum_2d_kernel(Z, X, BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, reduce_dim: tl.constexpr,
                               NON_REDUCE_DIM: tl.constexpr):
        offsets = tl.arange(0, BLOCK_0)[:, None] * BLOCK_1 + tl.arange(0, BLOCK_1)[None, :]
        vals = tl.load(X + offsets)
        z = tl.reduce(vals, reduce_dim, sanitize_add)
        tl.store(Z + tl.arange(0, NON_REDUCE_DIM), z)

    BLOCK_0 = 16
    BLOCK_1 = 32
    NON_REDUCE_DIM = BLOCK_1 if reduce_dim == 0 else BLOCK_0
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK_0, BLOCK_1], device="npu", dtype=torch.int32)
    Z = torch.zeros([NON_REDUCE_DIM], device="npu", dtype=torch.int32)
    sanitize_sum_2d_kernel[(1, )](Z, X, BLOCK_0=BLOCK_0, BLOCK_1=BLOCK_1, reduce_dim=reduce_dim,
                                  NON_REDUCE_DIM=NON_REDUCE_DIM)
    torch.testing.assert_close(Z, X.sum(reduce_dim).to(torch.int32))


@pytest.mark.interpreter
def test_dtype():

    device = "npu"

    @triton.jit
    def kernel(X):
        dtype_x: tl.constexpr = X.dtype.element_ty
        tl.static_assert(dtype_x == tl.int32)
        tl.static_assert(dtype_x == tl.constexpr(tl.int32))
        tl.static_assert(dtype_x == tl.int8 or (dtype_x == tl.int16 or dtype_x == tl.int32))

    X = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](X)


def test_side_effectful_scan():
    device = "npu"

    if device != "npu":
        pytest.skip()

    @triton.jit(debug=True)
    def sanitize_cumsum_kernel(Z, X, BLOCK: tl.constexpr):
        vals = tl.load(X + tl.arange(0, BLOCK))
        z = tl.associative_scan(vals, 0, sanitize_add)
        tl.store(Z + tl.arange(0, BLOCK), z)

    BLOCK = 512
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK], device="npu", dtype=torch.int32)
    X[:300] = 32
    X[300:] = 0
    Z = torch.zeros_like(X)
    sanitize_cumsum_kernel[(1, )](Z, X, BLOCK=BLOCK)
    torch.testing.assert_close(Z, X.cumsum(0).to(torch.int32))


# stress test slice layout usages in reductions.


@triton.jit
def gather_test_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                       src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr,
                       idx_dim1: tl.constexpr, idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                       out_dim0: tl.constexpr, out_dim1: tl.constexpr, out_stride0: tl.constexpr,
                       out_stride1: tl.constexpr):
    src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
    src = tl.load(src_ptr + src_offs)

    idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
    tl.store(out_ptr + out_offs, out)


@triton.jit
def gather_test_kernel_1d(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, idx_dim0: tl.constexpr,
                          out_dim0: tl.constexpr):
    src_offs = tl.arange(0, src_dim0)
    src = tl.load(src_ptr + src_offs)

    idx_offs = tl.arange(0, idx_dim0)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = tl.arange(0, out_dim0)
    tl.store(out_ptr + out_offs, out)


@triton.jit
def mul_jit_function(x, y):
    return x * y


@triton.jit
def apply_binary_op(x, combine_op):
    return combine_op(x, x)


def test_jit_function_arg():

    device = "npu"

    @triton.jit
    def square_kernel_jit_function(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        in_data = tl.load(in_ptr + offsets)
        out_data = apply_binary_op(in_data, mul_jit_function)  # pass a JITFunction into another JITFunction
        tl.store(out_ptr + offsets, out_data)

    BLOCK_SIZE = 16
    x = torch.full((BLOCK_SIZE, ), 3.0, device=device)
    out = torch.empty((BLOCK_SIZE, ), device=device)
    expect = torch.full((BLOCK_SIZE, ), 9.0, dtype=x.dtype, device=device)

    square_kernel_jit_function[(1, )](x, out, BLOCK_SIZE)

    torch.testing.assert_close(out, expect)


@pytest.mark.interpreter
def test_aliasing():

    device = "npu"

    @triton.jit
    def aliasing_kernel(buffer, buffer2):
        triton.language.store(buffer, 1)

    buffer = torch.zeros(1, device=device)
    aliasing_kernel[(1, )](buffer, buffer)
    assert buffer[0] == 1


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [
    d for d in map(tl.dtype, tl.dtype.SINT_TYPES + tl.dtype.UINT_TYPES + tl.dtype.STANDARD_FP_TYPES) if d != tl.float64
])
def test_dtype_tensor(dtype):

    device = "npu"

    @triton.jit
    def dtype_tensor_kernel(dtype: tl.constexpr):
        tensor = tl.zeros((1, ), dtype)

    dtype_tensor_kernel[(1, )](dtype)


@pytest.mark.interpreter
def test_short_circuiting():

    device = "npu"

    @triton.jit
    def short_circuiting_kernel(x):
        if (x is not None) and hasattr(x, "dtype") and isinstance(
                x.dtype, tl.pointer_type) and (x.dtype.element_ty == tl.int32) and (tl.load(x) > 42):
            tl.store(x, 42)

    def f(x):
        short_circuiting_kernel[(1, )](x, num_warps=1)

    f(None)  # should succeed with NoneType
    f(1)  # should succeed with tl.constexpr type
    f(2)  # should succeed with integer type

    def g(y, dtype):
        x = torch.full((1, ), y, device=device, dtype=dtype)
        f(x)
        return x.item()

    assert g(37.5, torch.float32) == 37.5
    assert g(84.0, torch.float32) == 84.0
    assert g(-76893, torch.int32) == -76893
    assert g(100000, torch.int32) == 42
    assert g(100000, torch.int64) == 100000


@pytest.mark.interpreter
@pytest.mark.filterwarnings("ignore:If conditional called with multidimensional Tensor*")
def test_unsplat():

    device = "npu"

    @triton.jit
    def unsplat_kernel(x, explicit: tl.constexpr):

        # this is a single-element tensor:
        condition = tl.load(x + tl.arange(0, 1)) > 42

        if explicit:
            condition = condition.item()

        if condition:
            tl.store(x, 42)

    def g(y, explicit):
        x = torch.full((1, ), y, device=device, dtype=torch.int32)
        unsplat_kernel[(1, )](x, explicit, num_warps=1)
        return x.item()

    assert g(41, False) == 41
    assert g(43, False) == 42
    assert g(41, True) == 41
    assert g(43, True) == 42


@pytest.mark.interpreter
def test_cumsum_dtype():

    device = "npu"

    @triton.jit
    def kernel(Z):
        x = tl.full((4, ), True, dtype=tl.int1)
        z = tl.cumsum(x, axis=0)
        tl.store(Z + tl.arange(0, 4), z)

    z = torch.zeros(4, dtype=torch.int32, device=device)
    kernel[(1, )](z)
    expected = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    assert torch.equal(z, expected)


@pytest.mark.interpreter
def test_tensor_member():

    device = "npu"

    @triton.jit
    def kernel():
        x = tl.arange(0, 16)
        tl.device_assert(tl.abs(x) == x.abs())
        tl.device_assert(tl.sum(x) == x.sum())

    kernel[(1, )]()


