import logging

import torch


class Diagonal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, offset, dim1, dim2):
        logging.debug("GEMS DIAGONAL")
        ctx.save_for_backward(inp)
        ctx.offset = offset
        ctx.dim1 = dim1
        ctx.dim2 = dim2
        return torch.diagonal(inp, offset, dim1, dim2)

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS DIAGONAL BACKWARD")
        (inp,) = ctx.saved_tensors
        grad_input = torch.zeros_like(inp)
        diag = torch.diagonal(grad_input, ctx.offset, ctx.dim1, ctx.dim2)
        diag.copy_(out_grad)
        return grad_input, None, None, None


def diagonal(inp, offset=0, dim1=0, dim2=1):
    return Diagonal.apply(inp, offset, dim1, dim2)
