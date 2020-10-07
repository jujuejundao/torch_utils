import sys
sys.path.append("../../src/")
from utils import *


class SaTasNet(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, nhead, num_layers, dim_feedforward, dropout):
        super(SaTasNet, self).__init__()
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C

        self.encoder = Encoder(L, N) # [M, N, K]
        self.pe_encoding = PositionalEncoding(N, dropout = 0.1)
        self.separator = cmEncoder(cmEncoderLayer(N,nhead = nhead,dim_feedforward = dim_feedforward, dropout=dropout),num_layers=num_layers)

        self.mask_conv1x1 = nn.Conv1d(N, C*N, 1, bias=False)
        self.decoder = Decoder(N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        mixture = self.encoder(mixture)
        M, N, K = mixture.size()
        
        # pad to *8 
        # est_source = F.pad(mixture, (0,1))
        est_source = est_source.transpose(0,1).transpose(0,2) 
        est_source = self.pe_encoding(est_source * math.sqrt(self.N))
        est_source = self.separator(est_source).transpose(0,1).transpose(1,2)
        est_source = self.mask_conv1x1(est_source)
        # est_source = est_source[:,:,:-1]
        est_source = est_source.view(M, self.C, N, K)

        est_source = self.decoder(mixture, est_source)
        
        return est_source


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self,mixture_w, est_mask):
        est_mask = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(est_mask, 2, 3) # [M, C, K, N]
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source


class cmEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(cmEncoder,self).__init__()
        self.layers = _clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, a_src):
        for mod in self.layers:
            a_src = mod(a_src)
        return a_src

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


    def forward(self, a_src):
        a_src2 = self.multihead_ca(a_src)

        a_src = a_src + self.a_dropout1(a_src2)
        a_src = self.a_norm1(a_src)
        a_src2 = self.a_linear2(self.a_dropout(F.relu(self.a_linear1(a_src))))
        a_src = a_src + self.a_dropout2(a_src2)
        a_src = self.a_norm2(a_src)

        return a_src

class multihead_ca(nn.Module):
    def __init__(self, d_model, nhead, att_dropout):
        super(multihead_ca, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_qkv = d_model // nhead
        self.a_linears = _clones(nn.Linear(d_model, d_model), 4)
        self.a_att_dropout = nn.Dropout(att_dropout)

    def forward(self, a_src):
        # [seq, batch, dim]
        nbatches = a_src.size(1)

        a_src = a_src.transpose(0,1) # [Batch, Seq, Dim]

        a_query, a_key, a_value = \
            [l(a).view(nbatches, -1, self.nhead, self.d_qkv).transpose(1, 2)
            for l, a in zip(self.a_linears, (a_src, a_src, a_src))]   # [batch, seq, head, dim] -> [batch, head, seq, dim]


        a_scores = torch.matmul(a_query, a_key.transpose(-1, -2)) / math.sqrt(self.d_qkv) # [batch, head, seq_q, seq_av_key]
        a_p_attn = F.softmax(a_scores, dim = -1)
        a_p_attn = self.a_att_dropout(a_p_attn)
        a = torch.matmul(a_p_attn, a_value).transpose(1,2).transpose(0,1) # [batch, head, seq, dim] -> [seq, batch, head, dim]
        a = a.contiguous().view(-1, nbatches, self.nhead * self.d_qkv)

        return self.a_linears[-1](a)
