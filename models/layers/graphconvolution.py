import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    简单的 GCN layer
    """

    def __init__(self, in_features, out_features, n_relations=1, K=1, bias=True):
        """_summary_

        Args:
        K=1 使用(T.Kipf and M.Welling, ICLR 2017) 论文的方法; K=2 使用(M.Defferrard, X.Bresson, and P.Vandergheynst, NIPS 2017)  论文的方法.
            in_features (_type_): 输入特征
            out_features (_type_): 输出特征
            bias (bool, optional): 偏置. Defaults to True.
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        # Parameter用于将参数自动加入到参数列表
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  # 为模型添加参数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # 最新spmm函数是在torch.sparse模块下，但是不能用
        # 使用稀疏矩阵乘法，
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
