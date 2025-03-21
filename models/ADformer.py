import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer,FFTDecomp,DECOMP, emd_decomp,my_Layernorm,vmd_decomp, SeriesDecompAttention, series_decomp,stl_decomp
from layers.RevIN import RevIN, ResidualMLP
from layers.Myattn import CT_MSA
from layers.SelfAttention_Family import AttentionLayer,FullAttention,ProbAttention


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trend_way=configs.trend_way

        # Decomp
        kernel_size = [4,8,12,16,20]
        seasonal_period=15
        if configs.decomp_way == 0:
            self.decomp = series_decomp(kernel_size=configs.moving_avg)
        elif configs.decomp_way==1:
            self.decomp=SeriesDecompAttention(kernel_size,num_heads=len(kernel_size))
        elif configs.decomp_way==2:
            self.decomp=stl_decomp(seasonal_period)
        elif configs.decomp_way==3:
            self.decomp=vmd_decomp(n_modes=5, alpha=2000, tau=0.0, K=5, DC=False, init=1, tol=1e-7)
        elif configs.decomp_way==4:
           self.decomp= emd_decomp(n_imfs=5)
        elif configs.decomp_way==5:
           self.decomp=FFTDecomp(cutoff=0.1)
        elif configs.decomp_way==6:
           self.decomp = DECOMP(ma_type='ema', alpha=0.3, beta=0.3)

        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        if configs.attn_way==1 :
            self_attn1 = CT_MSA(dim=configs.d_model, depth=3, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.seq_len, dropout=configs.dropout, device=self.device)
            self_attn2 = CT_MSA(dim=configs.d_model, depth=3, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.pred_len+configs.label_len, dropout=configs.dropout, device=self.device)
            cross_attn = CT_MSA(dim=configs.d_model, depth=3, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.pred_len+configs.label_len, dropout=configs.dropout, device=self.device)
        elif configs.attn_way==2 :
            self_attn1 = AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads)
            self_attn2 = AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads)
            cross_attn = AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads)
        elif configs.attn_way==3:
            self_attn1 =AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            self_attn2 =AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads)
            cross_attn =AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads)
        if configs.attn_way==4 :# danceng
            self_attn1 = DBCT_MSA(dim=configs.d_model, depth=1, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.seq_len, dropout=configs.dropout, device=self.device)
            self_attn2 = DBCT_MSA(dim=configs.d_model, depth=1, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.pred_len+configs.label_len, dropout=configs.dropout, device=self.device)
            cross_attn = DBCT_MSA(dim=configs.d_model, depth=1, heads=configs.n_heads, window_size=None, mlp_dim=configs.d_ff, num_time=configs.pred_len+configs.label_len, dropout=configs.dropout, device=self.device)
        if configs.attn_way==5 :# xishu
            self_attn1 =AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            self_attn2 =AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads)
            cross_attn =AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads)              
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # AutoCorrelationLayer(
                    #     AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                    #                     output_attention=configs.output_attention),
                    #     configs.d_model, configs.n_heads),
                    self_attn1,
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # AutoCorrelationLayer(
                    #     AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                    #                     output_attention=False),
                    #     configs.d_model, configs.n_heads),
                    # AutoCorrelationLayer(
                    #     AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                    #                     output_attention=False),
                    #     configs.d_model, configs.n_heads),
                    self_attn2,
                    cross_attn,
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.residual_mlp = ResidualMLP(configs.seq_len, configs.d_model, configs.pred_len)
        self.revin_trend = RevIN(configs.enc_in).to(self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init= self.decomp(x_enc)
        

        if self.trend_way==1:
        # trend
            trend_init = self.revin_trend(trend_init, 'norm')
            trend_out = self.residual_mlp(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.revin_trend(trend_out, 'denorm')
        elif self.trend_way==2 :
            trend_init = self.revin_trend(trend_init, 'norm')
            trend_out = self.trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.revin_trend(trend_out, 'denorm')
        else :
            trend_out = self.residual_mlp(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
       
        # enc
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, _ = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)

        # final
        dec_out =  seasonal_part[:, -self.pred_len:, :] + trend_out

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]