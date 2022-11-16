#!/usr/bin/env python3

import math
from math import floor
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
################################################################################
################################################################################



#==============================================================================#
#===============================Useful functions===============================#
#==============================================================================#
# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        # nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

def output_shape_conv2D_or_pooling2D(h_in, w_in, padding, dilation, stride, kernel_size, layer_type='conv2D'):
    # Putting the args under the right format (if they aren't)
    if (type(padding) == int):
        padding = (padding, padding)
    if (type(dilation) == int):
        dilation = (dilation, dilation)
    if (type(stride) == int):
        stride = (stride, stride)
    if (type(kernel_size) == int):
        kernel_size = (kernel_size, kernel_size)

    # Computing the output shape
    if (layer_type == 'conv2D' or layer_type == 'maxpool2D'):
        h_out = floor( ( h_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1 )/(stride[0]) + 1 )
        w_out = floor( ( w_in + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1 )/(stride[1]) + 1 )
    elif (layer_type == 'avgpool2D'):
        h_out = floor( ( h_in + 2*padding[0] - kernel_size[0] )/(stride[0]) + 1 )
        w_out = floor( ( w_in + 2*padding[1] - kernel_size[1] )/(stride[1]) + 1 )
    else:
        raise NotImplementedError('Layer type {} not supported'.format(layer_type))
    return h_out, w_out

#==============================================================================#
#=============================Positional Encoders=============================#
#==============================================================================#
class PositionalEncoding(nn.Module):
    """
        Modified version of the code in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        to put the batch first
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if (d_model % 2 == 1):
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1]) # We do not take the
            # last component because d_model is odd so (last_odd_index - 1) = second_to_last_even_index
            # For instance, if d_model = 125, we will have a list of indices
            # from 0 to 124, so the last even index is 123 and 123-1=122 and
            # 122 is the second to las even number
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)

#==============================================================================#
#===================================Encoders===================================#
#==============================================================================#

class TransformerEncoderMultichannelCNN(nn.Module):

    def __init__(
                    self,
                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    nb_features_projection: float = 50,
                    d_model: int = 64,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.classification_pool = classification_pool
        self.in_channels = in_channels

        # Layers
        # Input embedding layers
        self.n_conv_layers = n_conv_layers
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=self.d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)

        # Class token parameter (if used)
        if (self.classification_pool.lower() == 'classtoken'):
            self.class_token = nn.Parameter(torch.randn(self.d_model)) # Normal random tensor
        else:
            raise ValueError("Classification pooling mode {} is not valid".format(self.classification_pool))

        # Positional Encoder Layer
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
        """
        # Extraction of the embeddings
        x = F.leaky_relu(self.conv1(src))
        x = F.leaky_relu(self.conv2(x))
        for i in range(self.n_conv_layers):
          x = F.leaky_relu(self.conv(x))
          x = self.maxpool(x)
        x = torch.swapaxes(x, 1, 2) # Becuase the input of the PE and the Transformer
        # has to be under the format (batch_size, seq_len, feat_dim)

        # Class token if used
        if (self.classification_pool.lower() == 'classtoken'):
            batch_size = x.shape[0]
            class_tokens = self.class_token.expand(batch_size, -1)
            class_tokens = class_tokens.unsqueeze(dim=1)
            x = torch.cat([class_tokens, x], dim=1)
        else:
            raise ValueError("Classification pooling mode {} is not valid".format(self.classification_pool))

        # Positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        encoding = self.transformer_encoder(x)

        # Embedded representation
        if (self.classification_pool.lower() == 'classtoken'):
            class_hidden_features = encoding[:, 0]
        else:
            raise ValueError("Classification pooling mode {} is not valid".format(self.classification_pool))

        return class_hidden_features

class EncoderTimeFrequency2DCNN(nn.Module):
    def __init__(self, nb_init_filters=12, increase_nb_filters_mode='multiplicative', pooling_mode='maxpool', input_shape=(3, 214, 100)):
        super(EncoderTimeFrequency2DCNN, self).__init__()
        # Defining the height and width of the input
        input_channels, h_in, w_in = input_shape[0], input_shape[1], input_shape[2]

        # First pattern
        output_channels = nb_init_filters
        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, dilation=1) # Padding one to keep the same spatial dimension as the input
        self.batchNorm_1 = nn.BatchNorm2d(num_features=output_channels, eps=0.001, momentum=0.99)
        # Computing output size after convolution
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        # Computing the output size after the pooling layer (always applied after conv2D)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=0, dilation=1, stride=2, kernel_size=2, layer_type=(pooling_mode + '2D'))

        # Second pattern
        if (increase_nb_filters_mode == 'additive' or increase_nb_filters_mode == 'Additive'):
            input_channels = output_channels
            output_channels = output_channels + 10
        elif (increase_nb_filters_mode == 'multiplicative' or increase_nb_filters_mode == 'Multiplicative'):
            input_channels = output_channels
            output_channels = 2*output_channels
        else:
            raise NotImplementedError("Mode {} to increase the number of filters is not supported".format(increase_nb_filters_mode))
        self.conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) # TODO: CHANGE PADDING TO KEEP SAME DIMENSION
        self.batchNorm_2 = nn.BatchNorm2d(num_features=output_channels, eps=0.001, momentum=0.99)
        # Computing output size after convolution
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        # Computing the output size after the pooling layer (always applied after conv2D)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=0, dilation=1, stride=2, kernel_size=2, layer_type=(pooling_mode + '2D'))

        # Third pattern
        if (increase_nb_filters_mode == 'additive' or increase_nb_filters_mode == 'Additive'):
            input_channels = output_channels
            output_channels = output_channels + 10
        elif (increase_nb_filters_mode == 'multiplicative' or increase_nb_filters_mode == 'Multiplicative'):
            input_channels = output_channels
            output_channels = 2*output_channels
        else:
            raise NotImplementedError("Mode {} to increase the number of filters is not supported".format(increase_nb_filters_mode))
        self.conv_3 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) # TODO: CHANGE PADDING TO KEEP SAME DIMENSION
        self.batchNorm_3 = nn.BatchNorm2d(num_features=output_channels, eps=0.001, momentum=0.99)
        # Computing output size after convolution
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        # Computing the output size afterthe pooling layer (always applied after conv2D)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=0, dilation=1, stride=2, kernel_size=2, layer_type=(pooling_mode + '2D'))

        # Fourth pattern
        if (increase_nb_filters_mode == 'additive' or increase_nb_filters_mode == 'Additive'):
            input_channels = output_channels
            output_channels = output_channels + 10
        elif (increase_nb_filters_mode == 'multiplicative' or increase_nb_filters_mode == 'Multiplicative'):
            input_channels = output_channels
            output_channels = 2*output_channels
        else:
            raise NotImplementedError("Mode {} to increase the number of filters is not supported".format(increase_nb_filters_mode))
        self.conv_4 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) # TODO: CHANGE PADDING TO KEEP SAME DIMENSION
        self.batchNorm_4 = nn.BatchNorm2d(num_features=output_channels, eps=0.001, momentum=0.99)
        # Computing output size after convolution
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        # Computing the output size afterthe pooling layer (always applied after conv2D)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=0, dilation=1, stride=2, kernel_size=2, layer_type=(pooling_mode + '2D'))
        # Final values of h_in, w_in and output_channels
        self.h_in_final, self.w_in_final, self.output_channels_final = h_in, w_in, output_channels

        # Pooling layer
        if (pooling_mode == 'maxpool'):
            self.poolingLayer = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        elif (pooling_mode == 'avgpool'):
            self.poolingLayer = nn.AvgPool2d(kernel_size=2, stride=None, padding=0)
        else:
            raise NotImplementedError("Pooling mode {} not supported".format(pooling_mode))


    def forward(self, input):
        # First pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_1(self.conv_1(input))))
        # print("Data shape after first pattern: ", x.shape)

        # Second pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_2(self.conv_2(x))))
        # print("Data shape after second pattern: ", x.shape)

        # Third pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_3(self.conv_3(x))))
        # print("Data shape after third pattern: ", x.shape)


        # Fourth pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_4(self.conv_4(x))))

        # Reshape data to prepare it to the classification layers
        output = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        return output

class TransformerEncoderBimodal_RawAndSpec(nn.Module):

    def __init__(
                    self,

                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    d_model_raw: int = 64,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,

                    nb_init_filters: int = 12,
                    increase_nb_filters_mode: str = 'multiplicative',
                    pooling_mode: str = 'maxpool',
                    input_shape: list = (3, 214, 100),

                    fusion_strategy: str = 'Sum',

                    dim_common_space: int = 64
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model_raw = d_model_raw
        self.classification_pool = classification_pool
        self.in_channels = in_channels
        self.dim_common_space = dim_common_space
        self.fusion_strategy = fusion_strategy

        # Encoder model for the raw input
        self.raw_encoder = TransformerEncoderMultichannelCNN(
                                                            in_channels,
                                                            nhead,
                                                            d_hid,
                                                            nlayers,
                                                            dropout,
                                                            dim_common_space,
                                                            d_model_raw,
                                                            classification_pool,
                                                            n_conv_layers
                                                        )
        self.d_model_raw = self.raw_encoder.d_model

        # Encoder model for the spec representation
        self.spec_encoder = EncoderTimeFrequency2DCNN(
                                                        nb_init_filters,
                                                        increase_nb_filters_mode,
                                                        pooling_mode,
                                                        input_shape
                                                     )
        output_channels, h_in, w_in = self.spec_encoder.output_channels_final, self.spec_encoder.h_in_final, self.spec_encoder.w_in_final


        # Fusion layers
        if (self.fusion_strategy.lower() == 'weightedsumwithoutsoftmax'):
            #=======> First version: Sum attention weights without softmax of the weights
            self.attention_weight_raw = nn.Parameter(torch.randn(1))
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.attention_weight_spec = nn.Parameter(torch.randn(1))
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
        elif (self.fusion_strategy.lower() == 'weightedsumwithsoftmax'):
            # =======> Second version: Sum attention weights with softmax of the weights
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
            # Attention weights
            self.attention_weights = nn.Parameter(torch.randn(2))
        elif (self.fusion_strategy.lower() == 'multiplication'):
            # =======> Third version: Multiplication of embeddings
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
        elif (self.fusion_strategy.lower() == 'fclayerandmultiplication'):
            #=======> Fourth version: Adding additional FC Layer + Multiplication
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
            # Common projection layer
            self.layer_norm_fusion_common = nn.LayerNorm(dim_common_space)
            self.proj_common = nn.Linear(dim_common_space, dim_common_space)
        elif (self.fusion_strategy.lower() == 'sum'):
            # =======> Fifth version: Sum of the embeddings (without attention weights)
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
        elif (self.fusion_strategy.lower() == 'concatenation'):
            # =======> Sixth version: Concatenation
            # Raw input
            self.layer_norm_fusion_raw = nn.LayerNorm(self.d_model_raw)
            self.proj_raw = nn.Linear(self.d_model_raw, dim_common_space)
            # Time Frequency input
            self.layer_norm_fusion_spec = nn.LayerNorm(output_channels*h_in*w_in)
            self.proj_spec = nn.Linear(output_channels*h_in*w_in, dim_common_space)
        else:
            raise NotImplementedError('Fusion strategy {} has not been implemented.'.format(self.fusion_strategy))

    # def forward(self, src: dict) -> Tensor:
    def forward(self, src) -> Tensor:
        """
        Args:
        """
        # Getting the src for each modality
        assert (len(src) == 2) # If not, we have an unimodal or multimodal dataset
        # with more than two modalities
        for feature_type in src:
            if (feature_type.lower() == 'rawaudio') or (feature_type.lower() == 'rawsignal'):
                raw_src = src[feature_type]
            elif (feature_type.lower() == 'spectrogram'):
                spec_src = src[feature_type]
            else:
                raise ValueError("Feature type {} not valid for Bi-modal Transformer Spectrogram and Raw signal".format(feature_type))

        # Encoding using the Transformer model
        class_hidden_features_raw = self.raw_encoder(raw_src)
        class_hidden_features_spec = self.spec_encoder(spec_src)

        # Fusion layers
        projected_repr_raw = self.proj_raw(self.layer_norm_fusion_raw(class_hidden_features_raw))
        projected_repr_spec = self.proj_spec(self.layer_norm_fusion_spec(class_hidden_features_spec))
        if (self.fusion_strategy.lower() == 'weightedsumwithoutsoftmax'):
            #=======> First version: Sum attention weights without softmax of the weights
            class_hidden_features = self.attention_weight_raw*projected_repr_raw + self.attention_weight_spec*projected_repr_spec
        elif (self.fusion_strategy.lower() == 'weightedsumwithsoftmax'):
            #=======> Second version: Sum attention weights with softmax of the weights
            normalized_weights = F.softmax(self.attention_weights, dim=0)
            class_hidden_features = normalized_weights[0]*projected_repr_raw + normalized_weights[1]*projected_repr_spec
        elif (self.fusion_strategy.lower() == 'multiplication'):
            #=======> Third version: Multiplication of embeddings
            class_hidden_features = projected_repr_raw*projected_repr_spec
        elif (self.fusion_strategy.lower() == 'fclayerandmultiplication'):
            #=======> Fourth version: Adding additional FC Layer + Multiplication
            class_hidden_features = F.leaky_relu(self.proj_common(self.layer_norm_fusion_common(projected_repr_raw))*self.proj_common(self.layer_norm_fusion_common(projected_repr_spec)))
        elif (self.fusion_strategy.lower() == 'sum'):
            #=======> Fifth version: Sum of the embeddings (without attention weights)
            class_hidden_features = projected_repr_raw + projected_repr_spec
        elif (self.fusion_strategy.lower() == 'concatenation'):
            #=======> Sixth version: Concatenation
            class_hidden_features = torch.cat((projected_repr_spec, projected_repr_raw), 1)
        else:
            raise NotImplementedError('Fusion strategy {} has not been implemented.'.format(self.fusion_strategy))

        return class_hidden_features

#==============================================================================#
#==================================Classifier==================================#
#==============================================================================#

class TransformerClassifierBimodal_RawAndSpec(nn.Module):

    def __init__(
                    self,

                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    nb_features_projection: float = 50,
                    d_model_raw: int = 64,
                    num_classes: int = 3,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,

                    nb_init_filters: int = 12,
                    increase_nb_filters_mode: str = 'multiplicative',
                    pooling_mode: str = 'maxpool',
                    input_shape: list = (3, 214, 100),

                    fusion_strategy: str = 'Sum',

                    dim_common_space: int = 64
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model_raw = d_model_raw
        self.classification_pool = classification_pool
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dim_common_space = dim_common_space
        self.fusion_strategy = fusion_strategy

        # Encoder model for the raw input
        self.encoder = TransformerEncoderBimodal_RawAndSpec(
                                                                    in_channels,
                                                                    nhead,
                                                                    d_hid,
                                                                    nlayers,
                                                                    dropout,
                                                                    d_model_raw,
                                                                    classification_pool,
                                                                    n_conv_layers,
                                                                    nb_init_filters,
                                                                    increase_nb_filters_mode,
                                                                    pooling_mode,
                                                                    input_shape,
                                                                    fusion_strategy,
                                                                    dim_common_space
                                                            )


        # Classification layers
        if (self.fusion_strategy.lower() == 'concatenation'):
            # =======> Sixth version: Concatenation
            self.layer_norm_1 = nn.LayerNorm(self.dim_common_space*2) # For the CONCATENATION
            self.out_1 = nn.Linear(self.dim_common_space*2, nb_features_projection) # For the CONCATENATION
        else:
            # =======> Other versions
            self.layer_norm_1 = nn.LayerNorm(self.dim_common_space)
            self.out_1 = nn.Linear(self.dim_common_space, nb_features_projection)

        self.layer_norm_2 = nn.LayerNorm(nb_features_projection)
        self.out_2 = nn.Linear(nb_features_projection, self.num_classes)

    # def forward(self, src: dict) -> Tensor:
    def forward(self, src) -> Tensor:
        """
        Args:
        """
        # Multi-modal transformer encoder
        class_hidden_features = self.encoder(src)

        # Classification layers
        output = self.out_1(self.layer_norm_1(class_hidden_features))
        output = self.out_2(self.layer_norm_2(output))

        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


################################################################################
################################################################################
"""
    MAIN CLASS
"""
if __name__=='__main__':
    # Device to use for computation
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Creating the model
    in_channels = 1
    nhead = 4 # corresponds to the number of layers in the Encoder/Decoder
    d_hid = 64
    nlayers = 4
    dropout = 0.1
    nb_features_projection = 50
    d_model_raw = 64
    num_classes = 3
    classification_pool = 'ClassToken'
    n_conv_layers = 3
    nb_init_filters = 32
    increase_nb_filters_mode = "multiplicative"
    pooling_mode = "maxpool"
    input_shape = (1, 32, 39)
    fusion_strategy = 'concatenation'
    dim_common_space = 64
    model = TransformerClassifierBimodal_RawAndSpec(
                                                            in_channels,
                                                            nhead,
                                                            d_hid,
                                                            nlayers,
                                                            dropout,
                                                            nb_features_projection,
                                                            d_model_raw,
                                                            num_classes,
                                                            classification_pool,
                                                            n_conv_layers,
                                                            nb_init_filters,
                                                            increase_nb_filters_mode,
                                                            pooling_mode,
                                                            input_shape,
                                                            fusion_strategy,
                                                            dim_common_space
                                                        )
    model.to(device)

    # Named parameters
    # for name, param in model.named_parameters():
    #     print(name)
        # print(name, param)

    # Creating dummy data
    bs = 5
    # Dummy data representing the raw input
    audio_lenght = 188
    dummy_data_raw = torch.randn((bs, in_channels, audio_lenght)).to(device)
    print("Shape of the input raw data: {}".format(dummy_data_raw.shape))
    # Dummy data representing the Spec input
    dummy_data_spec = torch.randn((bs, input_shape[0], input_shape[1], input_shape[2])).to(device)
    # Final dummy data
    dummy_data = {
                    'RawAudio': dummy_data_raw,
                    'Spectrogram': dummy_data_spec
                 }

    # Summary of the model
    summary(model)

    # Testing the model with dummy data
    output = model(dummy_data)
    print("Output shape: ", output.shape)
