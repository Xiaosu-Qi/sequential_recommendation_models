import numpy as np

def hit_at_k(pred_items, true_item):
    return int(true_item in pred_items)

def ndcg_at_k(pred_items, true_item):
    if true_item in pred_items:
        index = pred_items.index(true_item)
        return 1 / np.log2(index + 2)
    return 0.0

def evaluate(model, dataloader, device, k=10):
    model.eval()
    hits, ndcgs = [], []

    for batch in dataloader:
        input_ids, target_pos = batch
        input_ids = input_ids.to(device)
        scores = model(input_ids)  # [B, seq_len, item_num]

        for i in range(len(input_ids)):
            # Get ground truth
            true_item = target_pos[i]
            scores_i = scores[i].detach().cpu().numpy()

            # Top-K ranking
            top_k = np.argsort(scores_i)[::-1][:k].tolist()

            hits.append(hit_at_k(top_k, true_item))
            ndcgs.append(ndcg_at_k(top_k, true_item))

    return np.mean(hits), np.mean(ndcgs)
