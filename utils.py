import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import manifold
from typing import Optional
from Weight import Weight
import copy
import os
import argparse

import random
def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def loss_adv(features, ad_net, logits=None):

    ad_out = ad_net(features, logits)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty

class LabelSmooth(nn.Module):
    """
    Label smooth cross entropy loss

    Parameters:
        - **num_class** (int): num of classes
        - **alpha** Optional(float): the smooth factor
        - **device** Optional(str): the used device "cuda" or "cpu"
    """

    def __init__(self,
                 num_class: int,
                 alpha: Optional[float] = 0.1,
                 device: Optional[str] = "cuda"):
        super(LabelSmooth, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.alpha) * targets + self.alpha / self.num_class
        loss = (-targets * log_probs).sum(dim=1)
        return loss.mean()

def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    r = torch.norm(Xs, dim=1)[0]
    Ct = r*Ct / (torch.norm(Ct, dim=1, keepdim=True)+1e-10)
    Cs = r*Cs / (torch.norm(Cs, dim=1, keepdim=True)+1e-10)

    K = Cs.size(0)
    # for each class
    for k in range(K):
        Xs_k = Xs[Ys==k]
        Xt_k = Xt[Yt==k]

        if len(Xs_k)==0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k,dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k,dim=0)
            else:
                Wt_k = Wt[Yt==k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)

        Cs[k, :] = (1-decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1-decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct

def robust_pseudo_loss(output,label,q=1.0):
    # weight[weight<0.5] = 0.0
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).cuda()
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1.0-torch.masked_select(output,mask)**q)/q
    return torch.sum(mae)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)
    
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse    

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x_hat, x):
        x_flat = x.view(x.shape[0], -1)
        x_hat_flat = x_hat.view(x_hat.shape[0], -1)
        
        # Compute the cosine similarity
        cos_sim = nn.functional.cosine_similarity(x_hat_flat, x_flat, dim=1)
        
        # Compute the loss
        loss = 1 - cos_sim
        
        # Average over the batch
        loss = torch.mean(loss)
        
        return loss
        
def generate_masks(tensor, mask_ratio=0.7):
    """
    Generate a different mask for each sample in the batch.

    Args:
    tensor (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of values to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """

    batch_size, num_channels, height, width = tensor.shape
    num_elements = height * width

    # Calculate the number of values to be masked in each sample
    num_values_to_mask = int(num_elements * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(tensor)

    # Iterate through the batch and create a mask for each sample
    for b in range(batch_size):
        for c in range(num_channels):
            # Generate random indices to mask
            indices_to_mask = torch.randperm(num_elements)[:num_values_to_mask]

            # Convert flat indices to 2D indices
            rows = indices_to_mask // width
            cols = indices_to_mask % width

            # Apply the mask
            masks[b, c, rows, cols] = 0
    masks = masks.view(batch_size, height, num_channels, width)
    return masks    

def generate_channel_masks(tensor, mask_ratio=0.7):
    """
    Generate a mask for each sample in the batch that masks out whole channels.

    Args:
    tensor (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of channels to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """
    batch_size, num_channels, height, width = tensor.shape

    # Calculate the number of channels to mask
    num_channels_to_mask = int(num_channels * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(tensor)

    # Iterate through the batch and create a mask for each sample
    for b in range(batch_size):
        # Generate random indices to mask channels
        channels_to_mask = torch.randperm(num_channels)[:num_channels_to_mask]

        # Apply the mask to the chosen channels
        masks[b, channels_to_mask, :, :] = 0
    masks = masks.view(batch_size, height, num_channels, width)
    return masks


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

def linear_mmd2( f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)#求矩阵的内积：即将矩阵内的元素依次点乘，然后再将所有的点乘结果相加，得到一位数的结果
        return loss

def marginal(source, target, kernel_type='rbf', kernel_mul=2.0, kernel_num=5,fix_sigma = None):
        if kernel_type == 'linear':
            return linear_mmd2(source, target)
        elif kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = guassian_kernel(
                source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def conditional(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(
            s_label, t_label, type='visual')
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss
    