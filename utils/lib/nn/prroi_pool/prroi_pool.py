import torch
import torch.nn as nn
import torch.autograd as ag

__all__ = ['PrRoIPool2D']

class PrRoIPool2DFunction(ag.Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), \
                'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())

        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)

        features = features.contiguous()
        rois = rois.contiguous()
        params = (pooled_height, pooled_width, spatial_scale)

        ctx.save_for_backward(features, rois)
        ctx.params = params

        output = torch.nn.functional.adaptive_max_pool2d(features, (pooled_height, pooled_width))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois = ctx.saved_tensors
        grad_input = grad_coor = None

        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = torch.zeros_like(features)
            grad_input = torch.nn.functional.adaptive_max_pool2d(grad_input, (features.size(2), features.size(3)))
            grad_input.scatter_add_(0, rois[:, 0].long(), grad_output)

        return grad_input, grad_coor, None, None, None

class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return PrRoIPool2DFunction.apply(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)