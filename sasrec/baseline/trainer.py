# sasrec/baseline/trainer.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        seqs, labels = batch
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(seqs)
        logits = output[:, -1, :]  # 只看最后一个位置
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
