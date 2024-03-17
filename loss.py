import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
# from torchgeometry.losses import DiceLoss as  dice_loss
# ------------------------------------------------------- #


def get_loss(loss_name: str, n_classes: int, thresh=0.7, weight=None, ignore_index=255):
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    elif loss_name == "softiou":
        return SoftIoULoss(n_classes)
    elif loss_name == "ohem":
        return OhemCrossEntropy2d(thresh=thresh, min_kept=100000, weight=weight)
    elif loss_name == "focal":
        return FocalLoss(alpha=0.5, gamma=2, weight=weight, ignore_index=ignore_index)
    elif loss_name == "softce":
        return SoftCrossEntropy()
    elif loss_name == "kl":
        return KlLoss()
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "jaccard":
        return jaccard_loss
    elif loss_name == "tversky":
        return tversky_loss
    else:
        raise ValueError("Invalid loss name.")


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(logit)

        pred = F.softmax(logit, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


# Adapted from OCNet Repository (https://github.com/PkuRainBow/OCNet)
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, thresh=0.6, min_kept=0, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_label = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
        """

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('hard ratio: {} = {} / {} '.format(round(len(valid_inds) / num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        print(np.sum(input_label != self.ignore_label))
        target = torch.from_numpy(input_label.reshape(target.size())).long().cuda()

        return self.criterion(predict, target)


class FocalLoss(nn.Module):
    """
    This function down-weights well-classified examples
    without giving same weight to training data in certain batch.
    The down-weighting is controlled by a hyperparameter gamma (0 <= gamma <= 5).
    This function is useful for imbalanced datasets and focuses training on hard examples.
    """
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, val_msk]
                lbl = lbl[:, val_msk]
                loss -= torch.mean(torch.mul(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0)))
            return loss / batch_size
        else:
            return torch.mean(torch.mul(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1)))


class KlLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, val_msk]
                lbl = lbl[:, val_msk]
                loss += torch.mean(F.kl_div(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0), reduction='none'))
            return loss / batch_size
        else:
            return torch.mean(F.kl_div(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1), reduction='none'))


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(logits)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1. - dice_score)


def dice_loss(scale=None):
    def fn(input, target):
        smooth = 1.

        if scale is not None:
            scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            iflat = scaled.view(-1)
        else:
            iflat = input.view(-1)

        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return fn


# Return Jaccard index, or Intersection over Union (IoU) value
def jaccard_loss(preds: Tensor, targs: Tensor, eps: float = 1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]

    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)

    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]

        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = F.softmax(preds, dim=1)

    true_1_hot = true_1_hot.type(preds.type())

    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)  # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
    return iou


def tversky_loss(true, logits, alpha=1, beta=1, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


if __name__ == "__main__":
    import numpy as np
    yt = np.random.random(size=(2, 1, 3, 3, 3))
    print(yt, '\n')

    yt = torch.from_numpy(yt)
    print(yt, '\n')

    yp = np.zeros(shape=(2, 1, 3, 3, 3))
    print(yp, '\n')
    yp = yp + 1
    print(yp, '\n')

    yp = torch.from_numpy(yp)
    print(yp, '\n')

    dl = DiceLoss()
    print(dl(yp, yt).item())

# ! --------------------------------------------------------------------------- !
