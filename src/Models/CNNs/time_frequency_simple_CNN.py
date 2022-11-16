"""
    Defines a simple (2D) CNN architecture to be trained on time-frequency
    representations of audio signals
"""
from math import floor

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

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

        # Second pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_2(self.conv_2(x))))

        # Third pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_3(self.conv_3(x))))


        # Fourth pattern
        x = self.poolingLayer(F.leaky_relu(self.batchNorm_4(self.conv_4(x))))

        # Reshape data to prepare it to the classification layers
        output = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        return output

class TimeFrequency2DCNN(nn.Module):
    def __init__(self, nb_init_filters=12, increase_nb_filters_mode='multiplicative', pooling_mode='maxpool', dropout_probability=0.2, input_shape=(3, 224, 96), num_classes=3):
        super(TimeFrequency2DCNN, self).__init__()
        # Defining the height and width of the input
        input_channels, h_in, w_in = input_shape[0], input_shape[1], input_shape[2]
        self.num_classes = num_classes

        # Encoder
        self.encoder = EncoderTimeFrequency2DCNN(
                                                    nb_init_filters,
                                                    increase_nb_filters_mode,
                                                    pooling_mode,
                                                    input_shape
                                                )

        # Fully-Connected Layer
        output_channels, h_in, w_in = self.encoder.output_channels_final, self.encoder.h_in_final, self.encoder.w_in_final
        self.fc = nn.Linear(in_features=output_channels*h_in*w_in, out_features=self.num_classes) # 3 out features because we have 3 classes

        # Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_probability)


    def forward(self, input):
        # Encoder
        x = self.encoder(input)

        # Fully Connected layer
        output = self.dropout(self.fc(x))

        return output

if __name__=='__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating dummy data
    nb_channels, h_in, w_in = 1, 32, 39
    data = torch.randn((1, nb_channels, h_in, w_in)).to(device)

    # Creating the model
    nb_init_filters = 32
    increase_nb_filters_mode = 'multiplicative'
    pooling_mode = 'maxpool'
    dropout_probability = 0.2
    input_shape = (1, 32, 39)
    num_classes = 5
    model = TimeFrequency2DCNN(
                                    nb_init_filters,
                                    increase_nb_filters_mode,
                                    pooling_mode,
                                    dropout_probability,
                                    input_shape,
                                    num_classes
                                ).to(device)

    # Summary of the model
    summary(model, (1, nb_channels, w_in, h_in))

    # # Parameters of the network
    # for name, param in model.named_parameters():
    #     print(name)

    # Evaluating the model
    output = model(data)
