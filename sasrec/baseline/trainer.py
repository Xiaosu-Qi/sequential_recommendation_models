import torch
import torch.nn as nn
import torch.optim as optim
from utils import sample_negative

def train(model, dataloader, device, epochs=10, lr=0.001, neg_samples=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_seq, pos_items = batch
            input_seq = input_seq.to(device)
            pos_items = pos_items.to(device)

            # Sample negative items for each positive item
            neg_items = sample_negative(pos_items, model.num_items, device, neg_samples)

            # Concatenate positive and negative items for loss computation
            # The shape of logits is [B, 1 + neg_samples]
            logits = model(input_seq, torch.cat([pos_items.unsqueeze(1), neg_items], dim=1))

            # The positive item is always at index 0
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
