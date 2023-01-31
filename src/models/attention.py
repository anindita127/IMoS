import torch
from torch import nn
from torch.nn import functional


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_embedding, d_k, d_v, num_out_joints=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_embedding, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_embedding, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_embedding, n_head * d_v, bias=False)
        self.w_combine = nn.Linear(n_head * d_v, d_embedding, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([num_out_joints, d_embedding], eps=1e-6)

    def forward(self, q, k, v, mask=None):
        b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(b, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(b, len_v, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
        attn = self.dropout(functional.softmax(attn, dim=-1))
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).contiguous().view(b, len_q, -1)
        z = self.dropout(self.w_combine(z))
        z = self.layer_norm(z + residual)  # Add & Norm
        return z, attn


class AttentionLayer(nn.Module):
    def __init__(self, d_embedding, n_head, d_k, d_v, num_out_joints, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_embedding, d_k, d_v, num_out_joints=num_out_joints, dropout=dropout)

    def forward(self, q, kv, mask=None):
        x, attn = self.attn(q, kv, kv, mask=mask)
        return x, attn


class AttentionModel(nn.Module):
    def __init__(self, dim_embedding, p_dropout, attention_params, num_out_joints):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            AttentionLayer(dim_embedding, attention_params['num_heads'],
                           attention_params['dim_k'], attention_params['dim_v'],
                           num_out_joints, dropout=p_dropout)
            for _ in range(attention_params['num_layers'])
        ])

    def forward(self, joints, poses, previous_joints=None, action_feature= None, mask=None, return_attns=False):
        attn_masks = []
        # Prepare Q
        q = joints
        # Prepare KV
        kv = [poses]
        if previous_joints is not None:
            kv.extend(previous_joints)
        if action_feature is not None:
            kv.append(action_feature)
        kv = torch.cat(kv, dim=1)
        # Run Attention
        z = q
        for layer in self.attention_layers:
            z, attn_mask = layer(z, kv, mask=mask)
            attn_masks += ([attn_mask] if return_attns else [])
        if return_attns:
            return z, attn_masks
        return z