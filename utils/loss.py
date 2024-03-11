import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# DiceLoss, BCEDiceLoss, FocalLoss, GDL(Generalized Dice Loss), HD Loss, BD Loss

# def flatten(tensor):
#     C = tensor.size(1)

#     axis_order = (1,0) +  tuple(range(2, tensor.dim()))
#     transposed = tensor.permute(axis_order)

#     return transposed.contiguous().view(C, -1)

# def compute_per_channel_dice(input, target, eplison=1e-6, weight=None):
#     assert input.size() == target.size()

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     denominator = (input).sum(-1) + (target).sum(-1)
#     return 2*(intersect / denominator.clamp(min=eplison))

# def compute_per_channel_iou(input, target, weight=None):
#     assert input.size() == target.size()

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     return (intersect / input.sum(-1) +  target.sum(-1) - intersect)

# class _AbstractDiceLoss(nn.Module):
#     def __init__(self, weight=None, normalization='sigmoid'):
#         super(_AbstractDiceLoss, self).__init__()
#         self.register_buffer('weight', weight)

#         assert normalization in ['sigmoid', 'softmax', 'none']
#         if normalization == 'sigmoid':
#             self.normalization = nn.Sigmoid()
#         elif normalization == 'softmax':
#             self.normalization = nn.Softmax(dim=1)
#         else:
#             self.normalization = lambda x: x

#     def dice(self, input, target, weight):
#         raise NotImplementedError
    
#     def forward(self, input, target):
#         # get probabilites from logits
#         input = self.normalization(input)

#         # computer per channel Dice coefficient
#         per_channel_dice = self.dice(input, target, weight=self.weight)

#         return 1. - torch.mean(per_channel_dice)

# # DiceLoss
# class DiceLoss(_AbstractDiceLoss):
#     def __init__(self, weight=None, normalization = 'sigmoid'):
#         super().__init__(weight, normalization)

#     def dice(self, input, target, weight):
#         return compute_per_channel_dice(input, target, weight=weight)
    
# IOULoss
# class IOULoss(_AbstractDiceLoss):
#     def __init__(self, weight=None, normalization= 'sigmoid'):
#         super().__init__(weight, normalization)

#     def dice(self, input, target, weight):
#         return compute_per_channel_iou(input, target, weight=weight)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class IOULoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)  
        
        num = torch.sum(torch.mul(predict, target), dim=1) 
        den = predict + target - num
        
        loss = 1 - num / den    
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))       

    
# BCEDiceLoss
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()
        self.iou = IOULoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target), self.beta * self.dice(input, target)

# from torch.nn.modules.loss import _Loss

# class DiceLoss(_Loss):
#     def __init__(self,
#             include_background: bool = True,
#             to_onehot_y: bool = False,
#             sigmoid: bool = True,
#             softmax: bool = False,
#             squared_pred: bool = True,
#             jaccard: bool = False,
#             smooth_nr: float = 1e-5,
#             smooth_dr: float = 1e-5,
#             batch: bool = False):
#         super().__init__()
#         self.include_background = include_background
#         self.to_onehot_y = to_onehot_y
#         self.sigmoid = sigmoid
#         self.softmax = softmax
#         self.squared_pred = squared_pred
#         self.jaccard = jaccard
#         self.smooth_nr = float(smooth_nr)
#         self.smooth_dr = float(smooth_dr)
#         self.batch = batch
#         self.register_buffer("class_weight", torch.ones(1))

#     def forward(self, input, target):
        
#         if self.sigmoid:
#             input = torch.sigmoid(input)

#         reduce_axis : list[int] = torch.arange(2, len(input.shape)).tolist()
#         intersection = torch.sum(target * input, dim = reduce_axis)

#         if self.squared_pred:
#             ground_o = torch.sum(target **2, dim=reduce_axis)
#             pred_o = torch.sum(input ** 2, dim=reduce_axis)
#         else:
#             ground_o = torch.sum(target, dim=reduce_axis)
#             pred_o = torch.sum(input, dim=reduce_axis)

#         denominator = ground_o + pred_o

#         if self.jaccard:
#             denominator = 2.0 * (denominator - intersection)

#         output = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

#         output = torch.mean(output)

#         return output
    
# class Loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bceloss = torch.nn.BCEWithLogitsLoss()
#         self.diceloss = DiceLoss()

#     def forward(self, input, target):
#         return self.bceloss(input, target) +  self.diceloss(input, target)