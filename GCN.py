import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class AGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, delta=0, drop=0.2):  # , layer=3, dropout=0.2, bias=False):

        super(AGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.delta = delta
        self.drop = drop

        # weight for cosine similarity
        self.weight = nn.Parameter(
            torch.FloatTensor(input_dim, output_dim).uniform_(-1 / np.sqrt(input_dim), 1 / np.sqrt(input_dim)))

        self.activation = nn.ReLU()

    def forward(self, e, device):
        # need to use dropout on W_l
        weight = F.dropout(self.weight, self.drop)

        # construct eq. 2 (A_i_j = cos(e_i W_{cos}, e_j W_{cos})) --> also equivalent to (H^l W^l)
        A_i_j = e.weight[1:,:] @ weight
        A_i_j_norm = F.normalize(A_i_j)
        A_i_j_norm = A_i_j_norm @ A_i_j_norm.T

        # apply a masking following eq. 3 in the paper
        # add small noise to not have zero values
        A_i_j_hat = A_i_j_norm * (A_i_j_norm > self.delta)

        # calculate diagonal matrix (D^(-0.5))
        D = 1 / (torch.sqrt(torch.sum(A_i_j_hat, dim=1, keepdim=True) + 1e-8))
        D_hat = torch.eye(A_i_j_hat.shape[0], A_i_j_hat.shape[1]).to(device) * D
        # update_adj = A_i_j / D / D.T

        # Eq. 7: 𝑨˜ = 𝑫ˆ(-0.5) 𝑨ˆ𝑫ˆ(-0.5))
        A_tilda = D_hat @ A_i_j_hat @ D_hat

        # Eq. 6: 𝑬^(𝑙+1) = 𝜎(𝑨𝑬˜^(𝑙))
        result = [e.weight[1:,:]]
        layer = self.activation(A_tilda @ e.weight[1:,:])
        result.append(layer)

        for counter in range(1, self.num_layers):
            layer = self.activation(A_tilda @ layer)
            result.append(layer)

        # apply sum pooling as described in 4.1.5 on the aggregation of graph representations
        result = torch.sum(torch.stack(result, dim=1), dim=1)

        result = torch.cat([e.weight[0, :].unsqueeze(dim=0), result], dim=0)

        return result, A_i_j_hat
