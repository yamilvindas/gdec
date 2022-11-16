#!/usr/bin/env python3

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchinfo import summary


################################################################################
################################################################################

#==============================================================================#
#==================================Classifier==================================#
#==============================================================================#
class SimpleAttentionLateFusion_Binary(nn.Module):

    def __init__(
                    self
                ):
        super().__init__()
        # Parameters of the model
        # self.softmax = torch.nn.Softmax(dim=0)
        self.attention_weight_class_1 = torch.nn.Parameter(torch.randn(2))
        self.attention_weight_class_2 = torch.nn.Parameter(torch.randn(2))

    def forward(self, src_1: Tensor, src_2: Tensor) -> Tensor:
        """
        Args:
        """
        # Attention weights
        self.tmp_attention_weight_class_1 = self.attention_weight_class_1
        self.tmp_attention_weight_class_2 = self.attention_weight_class_2
        bs = src_1.shape[0]

        # Class 1 output
        out_class_1 = (self.tmp_attention_weight_class_1[0]*src_1[:,0]+self.tmp_attention_weight_class_1[1]*src_2[:,0]).view((bs, 1))

        # Class 2 output
        out_class_2 = (self.tmp_attention_weight_class_2[0]*src_1[:,1]+self.tmp_attention_weight_class_2[1]*src_2[:,1]).view((bs, 1))

        # Final output
        output = torch.cat((out_class_1, out_class_2), axis=1)

        return output


class SimpleAttentionLateFusion_FiveClasses(nn.Module):

    def __init__(
                    self
                ):
        super().__init__()
        # Parameters of the model
        self.attention_weight_class_1 = torch.nn.Parameter(torch.randn(2))
        self.attention_weight_class_2 = torch.nn.Parameter(torch.randn(2))
        self.attention_weight_class_3 = torch.nn.Parameter(torch.randn(2))
        self.attention_weight_class_4 = torch.nn.Parameter(torch.randn(2))
        self.attention_weight_class_5 = torch.nn.Parameter(torch.randn(2))

    def forward(self, src_1: Tensor, src_2: Tensor) -> Tensor:
        """
        Args:
        """
        # Attention weights
        self.tmp_attention_weight_class_1 = self.attention_weight_class_1
        self.tmp_attention_weight_class_2 = self.attention_weight_class_2
        self.tmp_attention_weight_class_3 = self.attention_weight_class_3
        self.tmp_attention_weight_class_4 = self.attention_weight_class_4
        self.tmp_attention_weight_class_5 = self.attention_weight_class_5
        bs = src_1.shape[0]

        # Class 1 output
        out_class_1 = (self.tmp_attention_weight_class_1[0]*src_1[:,0]+self.tmp_attention_weight_class_1[1]*src_2[:,0]).view((bs, 1))

        # Class 2 output
        out_class_2 = (self.tmp_attention_weight_class_2[0]*src_1[:,1]+self.tmp_attention_weight_class_2[1]*src_2[:,1]).view((bs, 1))

        # Class 3 output
        out_class_3 = (self.tmp_attention_weight_class_3[0]*src_1[:,2]+self.tmp_attention_weight_class_3[1]*src_2[:,2]).view((bs, 1))

        # Class 4 output
        out_class_4 = (self.tmp_attention_weight_class_4[0]*src_1[:,3]+self.tmp_attention_weight_class_4[1]*src_2[:,3]).view((bs, 1))

        # Class 5 output
        out_class_5 = (self.tmp_attention_weight_class_5[0]*src_1[:,4]+self.tmp_attention_weight_class_5[1]*src_2[:,4]).view((bs, 1))


        # Final output
        output = torch.cat((out_class_1, out_class_2, out_class_3, out_class_4, out_class_5), axis=1)

        return output


################################################################################
################################################################################
"""
    MAIN CLASS
"""
if __name__=='__main__':
    # Device to use for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Creating the model
    nb_classes = 5
    if (nb_classes == 2):
        model = SimpleAttentionLateFusion_Binary()
    elif (nb_classes == 5):
        model = SimpleAttentionLateFusion_FiveClasses()
    else:
        raise ValueError("The number of classes nb_classes should be 2 or 5. Given value: {}".format(nb_classes))
    model.to(device)


    # Creating dummy data
    bs = 7
    # Dummy data
    dummy_data_1 = torch.randn((bs, nb_classes)).to(device)
    print("Shape of the input 1: {}".format(dummy_data_1.shape))
    dummy_data_2 = torch.randn((bs, nb_classes)).to(device)
    print("Shape of the input 1: {}".format(dummy_data_2.shape))

    # Testing the model with dummy data
    output = model(dummy_data_1, dummy_data_2)
    print("Output shape: ", output.shape)

    # Summary of the model
    summary(model)
