'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from resources.constants import hidden_layer_1_size, hidden_layer_2_size, no_output_classes, \
    neural_ntw_details, dropout


class gcn(nn.Module):
    """
    Class represents Graph Convolutional Network to handle dataset

    Attributes:
    -----------

      A_hat: torch.tensor
             Normalized symmetric adjacency matrix
     weight: nn.parameter.Parameter
             Weights of first layer
    weight2: nn.parameter.Parameter
             Weights of second layer
       bias: nn.parameter.Parameter
             Bias of first layer
      bias1: nn.parameter.Parameter
             Bias of second layer
        fc1: nn.Linear
             Linear transformation function for incoming data
    """

    def __init__(self, X_size, A_hat, bias=True):
        super(gcn, self).__init__()

        # Create A_hat with torch.tensor
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()

        # Initialize weights for the first layer of size (X_size, 330)
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, hidden_layer_1_size))  # Hidden Size 1
        # var = 2 / (self.weight.size(1) + self.weight.size(0)
        var = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.normal_(0, var)
        self.weight.data.uniform_(-var, var)

        # Initialize weights for the first layer of size (330, 130)
        self.weight2 = nn.parameter.Parameter(
            torch.FloatTensor(hidden_layer_1_size, hidden_layer_2_size))  # Hidden size 2
        var2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-var2, var2)
        '''var2 = 2 / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)'''

        # Initialize bias for both layers
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(hidden_layer_1_size))
            self.bias.data.normal_(0, var)
            self.bias1 = nn.parameter.Parameter(torch.FloatTensor(hidden_layer_2_size))
            self.bias1.data.normal_(0, var2)
        else:
            self.register_parameter()

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)

        # Create Linear Transformation function
        self.fc1 = nn.Linear(hidden_layer_2_size, no_output_classes)  # Hidden size 2 and no of classes
        table = []
        columns = ['A_hat shape', 'Weight 1 shape', 'Bias 1 shape', 'Weight 2 shape', 'Bias 2 shape', 'Output shape']
        details = [self.A_hat.shape, str(X_size) + str('X') + str(hidden_layer_1_size),
                   str(hidden_layer_1_size),
                   str(hidden_layer_1_size) + str('X') + str(hidden_layer_2_size),
                   str(hidden_layer_2_size),
                   str(hidden_layer_2_size) + str('X') + str(no_output_classes)]
        table.append(columns)
        table.append(details)

    ''' with open(neural_ntw_details, 'w') as csvfile:
         writer = csv.writer(csvfile)
         [writer.writerow(r) for r in table]'''

    def forward(self, X):
        """
        Forward propogation function for the Graph Convolutional Network
        :param X: (Identity Matrix)
        :return:
        """
        # Matrix multiplication of X (Identity Matrix) and weights at First layer
        X = torch.mm(torch.FloatTensor(X), self.weight)
        if self.bias is not None:
            X = X + self.bias
        ''' 
            Input Matrix multiplication of Normalized symmetric adjacency matrix 
            and newly calculated X to the relu function 
        '''
        # X = self.dropout(X)
        X = F.relu(torch.mm(self.A_hat, X))
        '''
            Matrix multiplication of first layer result with weights of second layer
        '''
        X = torch.mm(X, self.weight2)
        if self.bias1 is not None:
            X = X + self.bias1
        '''
            As we have to do multi-classification, 
            apply Softmax log to second layer result.
        '''
        # X = self.dropout(X)
        X = F.log_softmax(torch.mm(self.A_hat, X))
        return self.fc1(X)
