# sasrec/baseline/model.py
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = args.hidden_units
        self.max_seq_len = args.max_seq_len
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks

        self.item_emb = nn.Embedding(item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.hidden_units, nhead=self.num_heads)
            for _ in range(self.num_blocks)
        ])

        self.output_layer = nn.Linear(self.hidden_units, self.item_num)

    def forward(self, input_seq):
        seq_emb = self.item_emb(input_seq)
        pos = torch.arange(input_seq.size(1), device=input_seq.device).unsqueeze(0)
        seq_emb += self.pos_emb(pos)
        seq_emb = self.dropout(seq_emb)

        for block in self.attention_blocks:
            seq_emb = block(seq_emb)

        return seq_emb
