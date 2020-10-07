import sys
sys.path.append("../")
from utils import *
from ferNet import Fee_net

class TCAN(nn.Module):
    def __init__(self,
                    L=40,
                    N=512,
                    B=256,
                    nhead = 8,
                    dim_feedforward=512,
                    dropout= 0.1,
                    max_len= 5000,
                    nlayers=6,
                    visual_pretrain = 0):
        """
        Consist of an audio encoder
                        video encoder
                        av Bottleneck
                        cm encoder
                        decoder
        """
        super(TCAN, self).__init__()
        self.visual_pretrain = visual_pretrain
        self.L, self.N, self.B = L, N, B
        self.audioEncoder = audioEncoder(L=L, N=N)
        if not visual_pretrain:
            self.videoEncoder = Fee_net()
        self.avBottleneck = avBottleneck(N=N, B=B, dropout = dropout, max_len = max_len)
        self.cmEncoder = cmEncoder(cmEncoderLayer(B,nhead,dim_feedforward, dropout),nlayers)

        self.decoder = Decoder(N,B,L)
        self._reset_parameters()

    def forward(self, a_in, v_in):
        """
        a_in : [batch, second*sampling_rate]
        v_in : [batch, seq, dim]
        """

        a_in = self.audioEncoder(a_in) # a_in [batch, dim, seq]
        if not self.visual_pretrain:
            v_in = self.videoEncoder(v_in) # v_in [batch, seq, dim]
        a_out, v_out = self.avBottleneck(a_in, v_in) # all in [seq, batch, dim]
        a_out, v_out = self.cmEncoder(a_out, v_out) # all in [seq, batch, dim]
        a_out = a_out.transpose(0,1).transpose(1,2) # [batch, dim, seq]
        a_out = self.decoder(a_in,a_out)
        return a_out


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class audioEncoder(nn.Module):
    def __init__(self, L, N):
        super(audioEncoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, B, L):
        super(Decoder, self).__init__()
        self.B, self.L = B, L
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_mask = self.mask_conv1x1(est_mask) # [M, N, K]
        est_mask = F.relu(est_mask)

        source_w = mixture_w * est_mask  # [M, N, K]
        source_w = torch.transpose(source_w, 1, 2) # [M, K, N]

        est_source = self.basis_signals(source_w)  # [M, K, L]
        est_source = overlap_and_add(est_source,self.L//2) # M x C x T
        return est_source



class avBottleneck(nn.Module):
    """
    Perform layer normalisation and 1x1 convolution on audio input
    Perform a linear layer on visual input to match the dimension of audio input
    Perform positional encoding on visual and audio input
    """

    def __init__(self, N, B, dropout, max_len):
        super(avBottleneck, self).__init__()
        self.B = B
        self.a_layer_norm = ChannelwiseLayerNorm(N)
        self.a_bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        self.v_bottleneck = nn.Linear(512 , B)
        self.pe_encoding = PositionalEncoding(B, dropout)

    def forward(self, a_in, v_in):
        """
        a_in: [batch, dim, seq]
        v_in: [batch, seq, dim]
        """
        a_in = self.a_layer_norm(a_in) # [M, N, K] -> [M, N, K]
        a_in = self.a_bottleneck_conv1x1(a_in).transpose(0,1).transpose(0,2) # [M, N, K] -> [M, B, K] -> [seq, batch, dim]
        v_in = self.v_bottleneck(v_in).transpose(1,0) # [batch, seq, dim] -> [seq, batch, dim]

        a_in = self.pe_encoding(a_in * math.sqrt(self.B))
        v_in = self.pe_encoding(v_in * math.sqrt(self.B))

        return a_in, v_in


class cmEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(cmEncoder,self).__init__()
        self.layers = _clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, a_src, v_src):
        for mod in self.layers:
            a_src, v_src = mod(a_src, v_src)
        return a_src, v_src

def _clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class cmEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(cmEncoderLayer,self).__init__()
        self.multihead_ca = multihead_ca(d_model, nhead, att_dropout = dropout)

        self.a_linear1 = nn.Linear(d_model, dim_feedforward)
        self.a_dropout = nn.Dropout(dropout)
        self.a_linear2 = nn.Linear(dim_feedforward, d_model)
        self.a_norm1 = nn.LayerNorm(d_model)
        self.a_norm2 = nn.LayerNorm(d_model)
        self.a_dropout1 = nn.Dropout(dropout)
        self.a_dropout2 = nn.Dropout(dropout)

        self.v_linear1 = nn.Linear(d_model, dim_feedforward)
        self.v_dropout = nn.Dropout(dropout)
        self.v_linear2 = nn.Linear(dim_feedforward, d_model)
        self.v_norm1 = nn.LayerNorm(d_model)
        self.v_norm2 = nn.LayerNorm(d_model)
        self.v_dropout1 = nn.Dropout(dropout)
        self.v_dropout2 = nn.Dropout(dropout)

    def forward(self, a_src, v_src):
        a_src2, v_src2 = self.multihead_ca(a_src, v_src)

        a_src = a_src + self.a_dropout1(a_src2)
        a_src = self.a_norm1(a_src)
        a_src2 = self.a_linear2(self.a_dropout(F.relu(self.a_linear1(a_src))))
        a_src = a_src + self.a_dropout2(a_src2)
        a_src = self.a_norm2(a_src)

        v_src = v_src + self.v_dropout1(v_src2)
        v_src = self.v_norm1(v_src)
        v_src2 = self.v_linear2(self.v_dropout(F.relu(self.v_linear1(v_src))))
        v_src = v_src + self.v_dropout2(v_src2)
        v_src = self.v_norm2(v_src)

        return a_src, v_src

class multihead_ca(nn.Module):
    def __init__(self, d_model, nhead, att_dropout):
        super(multihead_ca, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_qkv = d_model // nhead
        self.a_linears = _clones(nn.Linear(d_model, d_model), 4)
        self.a_att_dropout = nn.Dropout(att_dropout)
        self.v_linears = _clones(nn.Linear(d_model, d_model), 4)
        self.v_att_dropout = nn.Dropout(att_dropout)

    def forward(self, a_src, v_src):
        # [seq, batch, dim]
        nbatches = a_src.size(1)

        a_src = a_src.transpose(0,1) # [Batch, Seq, Dim]
        v_src = v_src.transpose(0,1) # [Batch, Seq, Dim]

        a_query, a_key, a_value = \
            [l(a).view(nbatches, -1, self.nhead, self.d_qkv).transpose(1, 2)
            for l, a in zip(self.a_linears, (a_src, a_src, a_src))]   # [batch, seq, head, dim] -> [batch, head, seq, dim]
        v_query, v_key, v_value = \
            [l(v).view(nbatches, -1, self.nhead, self.d_qkv).transpose(1, 2)
            for l, v in zip(self.v_linears, (v_src, v_src, v_src))]


        av_key = torch.cat((a_key, v_key), 2)
        av_value = torch.cat((a_value, v_value), 2)

        a_scores = torch.matmul(a_query, av_key.transpose(-1, -2)) / math.sqrt(self.d_qkv) # [batch, head, seq_q, seq_av_key]
        a_p_attn = F.softmax(a_scores, dim = -1)
        a_p_attn = self.a_att_dropout(a_p_attn)
        a = torch.matmul(a_p_attn, av_value).transpose(1,2).transpose(0,1) # [batch, head, seq, dim] -> [seq, batch, head, dim]
        a = a.contiguous().view(-1, nbatches, self.nhead * self.d_qkv)

        
        v_scores = torch.matmul(v_query, av_key.transpose(-1, -2)) / math.sqrt(self.d_qkv)
        v_p_attn = F.softmax(v_scores, dim = -1)
        v_p_attn = self.v_att_dropout(v_p_attn)
        v = torch.matmul(v_p_attn, av_value).transpose(1,2).transpose(0,1)
        v = v.contiguous().view(-1, nbatches, self.nhead * self.d_qkv)

        return self.a_linears[-1](a), self.v_linears[-1](v)

if __name__ == '__main__':
    model = TCAN()
    a_in = torch.rand((2,48000))
    v_in = torch.rand((2,75,512))
    a_in = model(a_in, v_in)
    print(sum(p.numel() for p in model.parameters()))
    print(a_in.size())
