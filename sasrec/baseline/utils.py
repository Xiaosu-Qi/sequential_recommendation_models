import random
import torch

def get_attention_mask(seq):
    """
    padding mask for attention mechanism.
    """
    return (seq != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]


def sample_negative(item_pool, user_sequence, num_samples=1):
    """
    take negative samples from item_pool.
    """
    negatives = []
    while len(negatives) < num_samples:
        neg = random.choice(item_pool)
        if neg not in user_sequence:
            negatives.append(neg)
    return negatives


def get_causal_mask(seq_len):
    """
    Generate a causal mask for the attention mechanism.
    """
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
