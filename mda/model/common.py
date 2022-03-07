import torch.nn as nn
from torch.autograd import Function


class RevGrad(Function):
    """
    https://github.com/rpryzant/deconfounded-lexicon-induction
    """

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


class ReversalLayer(nn.Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.

        https://github.com/rpryzant/deconfounded-lexicon-induction
        """

        super().__init__()

    def forward(self, input_):
        return RevGrad.apply(input_)
