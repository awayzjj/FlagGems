import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    x_fp32 = x.to(tl.float32)
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    output = 0.5 * x_fp32 * (1 + erf(x_fp32 * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
        + 0.5 * erf(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x_fp32 * (
        (1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logging.debug("GEMS GELU FORWARD")
        if approximate == "tanh":
            out = gelu_tanh(A)
        else:
            out = gelu_none(A)
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS GELU BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            in_grad = gelu_backward_tanh(inp, out_grad)
        else:
            in_grad = gelu_backward_none(inp, out_grad)
        return in_grad, None


def gelu(A, *, approximate="none"):
    return Gelu.apply(A, approximate)


class InplaceGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logging.debug("GEMS GELU_ FORWARD")
        ctx.save_for_backward(A.clone())
        ctx.mark_dirty(A)
        ctx.approximate = approximate

        if approximate == "tanh":
            out = gelu_tanh(A, out0=A)
        else:
            out = gelu_none(A, out0=A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS GELU_ BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            in_grad = gelu_backward_tanh(inp, out_grad)
        else:
            in_grad = gelu_backward_none(inp, out_grad)
        return in_grad, None


def gelu_(A, *, approximate="none"):
    InplaceGelu.apply(A, approximate)
    return A
