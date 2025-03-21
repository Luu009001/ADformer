import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ResidualMLP(nn.Module):
    def __init__(self, seq_len, d_model, pred_len):
        super(ResidualMLP, self).__init__()
        self.fc1=nn.Linear(seq_len, d_model)
        self.act=nn.ReLU()
        self.fc2=nn.Linear(d_model, d_model)
        self.fc3=nn.Linear(d_model, pred_len)
        self.identity_transform = nn.Linear(seq_len, pred_len) if seq_len != pred_len else nn.Identity()

    def forward(self, x):
        batch_size, _, seq_len = x.size()  # 获取输入的维度
        x = x.view(batch_size, seq_len)  # 将输入从 [batch_size, 1, seq_len] 变为 [batch_size, seq_len]
        identity = self.identity_transform(x)  # 对输入进行变换，使其与输出维度一致
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        out += identity  # 残差连接
        out = out.view(batch_size, 1, -1)  # 将输出从 [batch_size, pred_len] 变为 [batch_size, 1, pred_len]
        return out
