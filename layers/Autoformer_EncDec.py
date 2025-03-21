import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL
import numpy as np
from vmdpy import VMD
from PyEMD import EMD  # 需要安装 PyEMD 库


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias



class stl_decomp(nn.Module):
    """
    STL decomposition block using LOESS-based trend and seasonal extraction.
    This block will decompose the input time series into trend, seasonal, and residual components.
    """
    def __init__(self, seasonal_period):
        super(stl_decomp, self).__init__()
        self.seasonal_period = seasonal_period

    def forward(self, x):
        # x: input time series (batch_size, channels, time_steps)
        x=x.permute(0,2,1)
        # Convert to numpy for STL (this is just for demonstration purposes, 
        # in practice you'd want to handle this efficiently)
        batch_size, channels, time_steps =x.shape
        trend_list = []
        seasonal_list = []
        residual_list = []
        
        # Decompose each series in the batch
        for i in range(batch_size):
            for j in range(channels):
                device = x.device  
                series = x[i, j, :].cpu().detach().numpy()
                
                # Perform STL decomposition on the series
                stl = STL(series, 15)
                result = stl.fit()

                trend = torch.tensor(result.trend, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                seasonal = torch.tensor(result.seasonal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                residual = torch.tensor(result.resid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                trend_list.append(trend)
                seasonal_list.append(seasonal)
                residual_list.append(residual)

        # Stack to get output in correct shape (batch_size, channels, time_steps)
        trend_output = torch.cat(trend_list, dim=0).view(batch_size, channels, time_steps).permute(0,2,1)
        seasonal_output = torch.cat(seasonal_list, dim=0).view(batch_size, channels, time_steps).permute(0,2,1)
        residual_output = torch.cat(residual_list, dim=0).view(batch_size, channels, time_steps).permute(0,2,1)


        return seasonal_output+residual_output,trend_output

class vmd_decomp(nn.Module):
    """
    VMD decomposition block using variational mode decomposition (VMD).
    This block will decompose the input time series into intrinsic mode functions (IMFs).
    """
    def __init__(self, n_modes, alpha=2000, tau=0.0, K=5, DC=False, init=1, tol=1e-7):
        """
        Initialize the VMD decomposition class.

        :param n_modes: Number of intrinsic modes to decompose the signal into.
        :param alpha:  Bandwidth constraint (higher values result in a more selective decomposition).
        :param tau:  Time-domain scaling factor.
        :param K: Number of modes to extract.
        :param DC: Whether to use a DC component.
        :param init: Initialization method for the modes.
        :param tol: Stopping tolerance for the algorithm.
        """
        super(vmd_decomp, self).__init__()
        self.n_modes = n_modes  # Number of IMFs to decompose the signal into
        self.alpha = alpha  # Bandwidth constraint
        self.tau = tau  # Time-domain scaling
        self.K = K  # Number of modes
        self.DC = DC  # Whether to use DC component
        self.init = init  # Initialization method for modes
        self.tol = tol  # Tolerance for stopping criterion

    def forward(self, x):
        """
        Forward pass for VMD decomposition.

        :param x: Input time series of shape (batch_size, time_steps, channels)
        :return: Trend, Seasonal components as tensors
        """
        x = x.permute(0, 2, 1)  # 转置为 (batch_size, channels, time_steps)
        batch_size, channels, time_steps = x.shape

        # List to store the decomposed IMFs
        modes_list = []

        # Decompose each series in the batch and each channel
        for i in range(batch_size):
            for j in range(channels):
                device = x.device
                series = x[i, j, :].cpu().detach().numpy()

                # Perform VMD decomposition on the series
                u, u_hat, omega = VMD(series, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)

                # Convert the decomposed modes back to tensors
                modes = [torch.tensor(mode, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) for mode in u]

                modes_list.append(modes)

        # Stack the results to get the output in the correct shape (batch_size, time_steps, channels)
        modes_output = []
        for i in range(self.n_modes):
            mode_output = torch.cat([mode[i] for mode in modes_list], dim=0).view(batch_size, channels, time_steps).permute(0, 2, 1)
            modes_output.append(mode_output)

        # 提取趋势项和季节项，残差包含在季节项中
        trend_output = modes_output[0]  # First mode is trend
        seasonal_output = torch.cat(modes_output[1:], dim=0).view(batch_size, time_steps, -1) if self.n_modes > 1 else torch.zeros_like(trend_output)  # Second mode is seasonal, includes residuals
        
        # 将seasonal_output的通道数压缩成 1
        seasonal_output = seasonal_output.mean(dim=-1, keepdim=True)  # 使用均值或其他操作压缩通道维度

        # Return the trend and seasonal components (seasonal contains residual as well)
        return seasonal_output, trend_output

class emd_decomp(nn.Module):
    """
    Empirical Mode Decomposition (EMD) block.
    This block decomposes input time series into intrinsic mode functions (IMFs) using EMD.
    """
    def __init__(self, n_imfs):
        """
        Initialize the EMD decomposition class.

        :param n_imfs: Number of intrinsic mode functions (IMFs) to decompose the signal into.
        """
        super(emd_decomp, self).__init__()
        self.n_imfs = n_imfs  # Number of IMFs to extract

    def forward(self, x):
        """
        Forward pass for EMD decomposition.

        :param x: Input time series of shape (batch_size, time_steps, channels)
        :return: Seasonal (sum of IMFs), Trend (last IMF)
        """
        x = x.permute(0, 2, 1)  # 转置为 (batch_size, channels, time_steps)
        batch_size, channels, time_steps = x.shape

        # List to store the decomposed IMFs
        imfs_list = []

        # Decompose each series in the batch and each channel
        for i in range(batch_size):
            for j in range(channels):
                device = x.device
                series = x[i, j, :].cpu().detach().numpy()

                # Perform EMD decomposition on the series
                emd = EMD()
                imfs = emd(series)  # Shape: (n_imfs, time_steps)

                # Ensure we have the expected number of IMFs
                if imfs.shape[0] < self.n_imfs:
                    # If not enough IMFs are extracted, pad with zeros
                    pad_size = self.n_imfs - imfs.shape[0]
                    imfs = np.vstack((imfs, np.zeros((pad_size, time_steps))))

                # Convert the decomposed IMFs back to tensors
                imfs_tensor = [torch.tensor(imf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) for imf in imfs[:self.n_imfs]]
                imfs_list.append(imfs_tensor)

        # Stack the results to get the output in the correct shape (batch_size, time_steps, channels)
        imfs_output = []
        for i in range(self.n_imfs):
            imf_output = torch.cat([imfs[i] for imfs in imfs_list], dim=0).view(batch_size, channels, time_steps).permute(0, 2, 1)
            imfs_output.append(imf_output)

        # Extract the trend and seasonal components
        trend_output = imfs_output[-1]  # Last IMF is considered the trend
        seasonal_output = sum(imfs_output[:-1])  # Sum of the first (n_imfs-1) IMFs

        # Return the seasonal component (including residuals) and trend
        return seasonal_output, trend_output

#5
class FFTDecomp(nn.Module):
    def __init__(self, cutoff=0.1):
        """
        :param cutoff: 低频部分的比例 (0~1)，决定趋势项
        """
        super(FFTDecomp, self).__init__()
        self.cutoff = cutoff

    def forward(self, x):
        """
        :param x: (batch_size, time_steps, channels)
        :return: trend_output, seasonal_output
        """
        device = x.device  # 确保所有变量都在相同设备上
        batch_size, time_steps, channels = x.shape

        x_fft = torch.fft.fft(x, dim=1)  # 对时间轴做 FFT

        # 计算频率索引并移动到 x 的设备
        freqs = torch.fft.fftfreq(time_steps, device=device)

        # 创建掩码
        mask = (freqs.abs() < self.cutoff).to(device)  # 低频部分为趋势
        mask = mask.view(-1, 1)  # 调整维度以进行广播

        # 分离趋势项（低频）和季节项（高频）
        trend_fft = x_fft * mask  # 低频信号
        seasonal_fft = x_fft * (~mask)  # 高频信号

        # 逆 FFT 变换回时域
        trend_output = torch.fft.ifft(trend_fft, dim=1).real
        seasonal_output = torch.fft.ifft(seasonal_fft, dim=1).real

        return seasonal_output, trend_output

#6
class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        seq_length = query.shape[1]

        # Split embedding into multiple heads
        values = values.view(N, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim)
        queries = query.view(N, seq_length, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.nn.functional.softmax(energy, dim=3)
        attention = self.dropout(attention)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, seq_length, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        return out
class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class SeriesDecompAttention(nn.Module):
    """
    使用多头自注意力机制进行序列分解的模块
    """
    def __init__(self, kernel_sizes, num_heads):
        super(SeriesDecompAttention, self).__init__()
        embed_size = len(kernel_sizes)  # 确保 embed_size 是 num_heads 的倍数
        self.moving_avgs = nn.ModuleList([MovingAvg(kernel_size, stride=1) for kernel_size in kernel_sizes])
        self.attention = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        if dim != 1:
            raise ValueError("Input x should have the third dimension of size 1")

        moving_means = []
        
        for moving_avg in self.moving_avgs:
            moving_mean = moving_avg(x)  # 得到每个移动平均窗口的输出
            moving_means.append(moving_mean)
        
        moving_means = torch.cat(moving_means, dim=-1)  # [batch_size, seq_len, 1, num_kernels]
        
        # 使用自注意力机制生成动态权重
        attention_output = self.attention(moving_means, moving_means, moving_means, mask=None)
        weights = torch.nn.functional.softmax(attention_output, dim=-1)
        
        # 确保权重的维度与 moving_means 匹配
        weights = weights.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_kernels]
        
        # 计算加权移动均值
        moving_mean = torch.sum(moving_means * weights, dim=-1)  # [batch_size, seq_len, 1]

        # 确保维度匹配
        moving_mean = moving_mean.unsqueeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len, 1]
        
        trend = moving_mean  # 设定移动均值为趋势成分
        seasonal = x - moving_mean  # 剩余部分作为季节性成分
        
        return seasonal, trend

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
