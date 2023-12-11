import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    # features: [bsz, f_dim]
    # better be L2 normalized in f_dim dimension
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, f_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [bsz, f_dim]')

        # L2 norm
        features = F.normalize(features)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[0] # every batch is a sample
        contrast_feature = features

        # force to use all mode
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count).mean()

        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    ref: https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2 + 1e-6) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class TripletLoss(nn.Module):

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        
        self.margin = margin

    def forward(self, features, labels):
        return self.batch_hard_triplet_loss(labels, features, self.margin)

    def pairwise_distances(self, z, squared = False):
        embeddings = torch.flatten(z, start_dim = 1)

        dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

        square_norm = torch.diagonal(dot_product)

        distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

        # zero tensor
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        zeros = Tensor(np.zeros(distances.size()))

        distances = torch.maximum(distances, zeros)

        if not squared:
            mask = torch.eq(distances, zeros).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)

        return distances

    def batch_hard_triplet_loss(self, labels, z, margin, squared=False):
        # zero tensor
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        zeros = Tensor(np.zeros((1, labels.size(0))))

        pairwise_dist = self.pairwise_distances(z, squared=squared)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels).float()
        anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist, _ = torch.max(anchor_positive_dist, 1)

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist, _ = torch.max(pairwise_dist, 1)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, 1)

        triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, zeros)

        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

    def get_anchor_positive_triplet_mask(self, labels):
        # zero tensor
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        zeros = Tensor(np.zeros((labels.size(0), labels.size(0))))

        indices_equal = torch.gt(torch.eye(labels.size(0)).cuda(), zeros)
        indices_not_equal = torch.logical_not(indices_equal)

        labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        mask = torch.logical_and(indices_not_equal, labels_equal)

        return mask

    def get_anchor_negative_triplet_mask(self, labels):
        labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        mask = torch.logical_not(labels_equal)

        return mask