from torch.nn import CrossEntropyLoss

from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)


class LigerFusedLinearCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerFusedLinearCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, lin_weight, _input, target, bias=None, final_logit_softcap_params=None):
        return LigerFusedLinearCrossEntropyFunction.apply(
            _input, lin_weight, target, bias, final_logit_softcap_params, self.ignore_index
        )
