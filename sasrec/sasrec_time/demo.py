import torch
from torch.utils.data import Dataset, DataLoader
from sasrec_time.model import SASRecTime
from sasrec_time.trainer import SASRecTimeTrainer
import random

# ---- Step 1: Toy Dataset ----
class ToyTimeDataset(Dataset):
    def __init__(self, num_users=100, seq_len=50, item_num=1000):
        self.data = []
        for _ in range(num_users):
            items = [random.randint(1, item_num) for _ in range(seq_len)]
            time_gaps = [random.randint(0, 30) for _ in range(seq_len)]
            label = random.choice(items)  # for simplicity, predict one from sequence
            self.data.append({
                "seq_items": torch.tensor(items, dtype=torch.long),
                "time_gaps": torch.tensor(time_gaps, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---- Step 2: Dataloader ----
def get_loader(batch_size=32):
    dataset = ToyTimeDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# ---- Step 3: Run Training ----
def main():
    item_num = 1000
    max_len = 50

    train_loader = get_loader(batch_size=32)
    val_loader = get_loader(batch_size=32)

    model = SASRecTime(item_num=item_num, max_len=max_len)
    trainer = SASRecTimeTrainer(model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(num_epochs=5)

if __name__ == "__main__":
    main()
