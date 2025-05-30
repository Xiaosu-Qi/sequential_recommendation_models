import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class SASRecTimeTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=0.001, weight_decay=0.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            seq_items = batch['seq_items'].to(self.device)         # [B, L]
            time_gaps = batch['time_gaps'].to(self.device)         # [B, L]
            labels = batch['labels'].to(self.device)               # [B]

            logits = self.model(seq_items, time_gaps)              # [B, item_num+1]
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, top_k=10):
        self.model.eval()
        total_hits = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                seq_items = batch['seq_items'].to(self.device)
                time_gaps = batch['time_gaps'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(seq_items, time_gaps)          # [B, item_num+1]
                top_k_preds = torch.topk(logits, k=top_k, dim=-1).indices  # [B, top_k]

                hits = (top_k_preds == labels.unsqueeze(1)).sum().item()
                total_hits += hits
                total_samples += labels.size(0)

        hit_rate = total_hits / total_samples
        return hit_rate

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_hit = self.evaluate()

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Hit@10 = {val_hit:.4f}")
