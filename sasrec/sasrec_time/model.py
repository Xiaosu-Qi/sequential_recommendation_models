import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecTime(nn.Module):
    def __init__(self, item_num, max_len, d_model=50, n_heads=1, n_layers=2, dropout=0.2, max_time_gap=100):
        super(SASRecTime, self).__init__()
        self.item_embedding = nn.Embedding(item_num + 1, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.time_embedding = nn.Embedding(max_time_gap + 1, d_model)

        self.attention_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout),
            num_layers=n_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, item_num + 1)

    def forward(self, seq_items, time_gaps):
        item_emb = self.item_embedding(seq_items)
        pos_ids = torch.arange(seq_items.size(1), device=seq_items.device).unsqueeze(0).expand(seq_items.size(0), -1)
        pos_emb = self.pos_embedding(pos_ids)
        time_emb = self.time_embedding(time_gaps.clamp(0, self.time_embedding.num_embeddings - 1))

        seq_emb = item_emb + pos_emb + time_emb
        seq_emb = self.layernorm(self.dropout(seq_emb))

        attn_mask = (seq_items == 0)  # [B, L]
        attn_output = self.attention_layers(seq_emb.transpose(0, 1), src_key_padding_mask=attn_mask)
        attn_output = attn_output.transpose(0, 1)  # [B, L, d_model]

        final_state = attn_output[:, -1, :]  # use last position
        logits = self.output_layer(final_state)
        return logits

