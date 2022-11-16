#!/usr/bin/env python3
"""
    This codes implements the Deep Embedding Clustering loss of the paper
    "Unsupervised Deep Embedding for Clustering Analysis" (Xie et al. 2015)
    This implementation is strongly inspired from the implementation
    https://github.com/vlukiyanov/pt-dec
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from typing import Optional

################################################################################
# Defining some tools
################################################################################
class ClusterAssignment(nn.Module):
    """
        Code from https://github.com/vlukiyanov/pt-dec
    """
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Code from https://github.com/vlukiyanov/pt-dec
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1.0) / 2.0
        numerator = numerator ** power
        # return numerator.t() / torch.sum(numerator, dim=1).t()
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Code from https://github.com/vlukiyanov/pt-dec
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return weight / torch.sum(weight, dim=1, keepdim=True)

################################################################################
# Defininf the DEC Loss
################################################################################

class DECLoss(nn.Module):
    """
        Code from https://github.com/vlukiyanov/pt-dec (some modifications were
        done by me to adapt it to our case)
    """
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        alpha: float = 1.0,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.
        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DECLoss, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )
        self.loss_function = nn.KLDivLoss(size_average=False)

    def forward(self, encoder: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        :param encoder: encoder to use
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        output = self.assignment(encoder(batch)) # computation of Q
        target = target_distribution(output) # computation of P
        dec_loss = self.loss_function(torch.log(output), target) / output.shape[0]
        return dec_loss
