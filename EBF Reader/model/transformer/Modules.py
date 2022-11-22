import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, sent_att=None, attn_mask=None, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # attn = attn.masked_fill(mask == 0, -2 ** 15)
            attn = attn.masked_fill(mask == 0, -1e9)
        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1)
        #     attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        if sent_att is not None:
            # g = 0.8
            # attn = g * attn + (1 - g) * (sent_att[:, None, None])
            attn = attn * (sent_att[:, None, None])
            attn = attn/(attn.sum(-1).unsqueeze(-1))
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
