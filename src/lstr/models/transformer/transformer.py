import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout

        # Place holder for online inference
        self.k_weights_cached = None
        self.k_pos_weights_cached = None

    def online_inference(self, q, k, v, k_pos, v_pos, attn_mask=None):
        assert (self.k_weights_cached is None) == (self.k_pos_weights_cached is None)
        if self.k_weights_cached is not None:
            k_weights_new = torch.bmm(q, k[:, [-1]].transpose(1, 2))
            k_weights = torch.cat((self.k_weights_cached[:, :, 1:], k_weights_new), dim=-1)
            k_pos_weights = self.k_pos_weights_cached
        else:
            k_weights = torch.bmm(q, k.transpose(1, 2))
            k_pos_weights = torch.bmm(q, k_pos.transpose(1, 2))
            self.k_weights_cached = k_weights
            self.k_pos_weights_cached = k_pos_weights
        attn_output_weights = k_weights + k_pos_weights

        if attn_mask is not None:
            attn_output_weights += attn_mask

        # import pdb; pdb.set_trace()
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, (v + v_pos))
        return attn_output, attn_output_weights

    def forward(self, q, k, v, attn_mask=None):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)
        return attn_output, attn_output_weights


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        self.dotproductattention = DotProductAttention(dropout)

        # Place holder for online inference
        self.q_cached = None
        self.k_cached = None
        self.v_cached = None
        self.k_pos_cached = None
        self.v_pos_cached = None

    def online_inference(self, q, k, v, pos, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if self.q_cached is not None:
            q = self.q_cached
        else:
            _b = self.in_proj_bias
            _start = None
            _end = embed_dim
            _w = self.in_proj_weight[:_end, :]
            if _b is not None:
                _b = _b[:_end]
            q = F.linear(q, _w, _b)
            self.q_cached = q

        assert (self.k_cached is None) == (self.k_pos_cached is None)
        if self.k_cached is not None:
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            k_new = F.linear(k[[0]], _w, None)
            k = torch.cat((self.k_cached[1:], k_new), dim=0)
            k_pos = self.k_pos_cached
            self.k_cached = k
        else:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(k, _w, None)
            k_pos = F.linear(pos, _w, _b)
            self.k_cached = k
            self.k_pos_cached = k_pos

        assert (self.v_cached is None) == (self.v_pos_cached is None)
        if self.v_cached is not None:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, _b)
            v_pos = self.v_pos_cached
            self.v_cached = v
        else:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, None)
            v_pos = F.linear(pos, _w, _b)
            self.v_cached = v
            self.v_pos_cached = v_pos

        q = q * scaling

        # import pdb; pdb.set_trace()
        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k_pos = k_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v_pos = v_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output, attn_weights = self.dotproductattention.online_inference(q, k, v, k_pos, v_pos, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), attn_weights

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output, attn_weights = self.dotproductattention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), attn_weights


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', custom_encoder=None, custom_decoder=None,output_attn_weights=False):
        super(Transformer, self).__init__()

        self.output_attn_weights = output_attn_weights

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')

        memory, enc_attn_weights = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output, dec_attn_weights = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        if self.output_attn_weights:
            return output, (enc_attn_weights, dec_attn_weights)
        else:
            return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, output_attn_weights=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        all_attn_weights = []
        for i, mod in enumerate(self.layers):
            output, attn_weights = mod(output, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask)
            all_attn_weights.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, all_attn_weights


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def online_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod.online_inference(output, memory, pos,
                                          tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        all_attn_weights = []

        for i, mod in enumerate(self.layers):
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            all_attn_weights.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, all_attn_weights


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, src2_attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, src2_attn_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def online_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, sa_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.multihead_attn.online_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, (sa_attn_weights, cross_attn_weights)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, sa_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, (sa_attn_weights, cross_attn_weights)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))
