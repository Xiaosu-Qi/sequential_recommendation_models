import torch
from torch.utils.data import DataLoader

from model import SASRec
from trainer import train
from utils import sample_negative
from evaluate import evaluate

num_users = 1000
num_items = 5000
max_seq_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_dummy_dataset(num_users=1000, max_len=50):
    import random
    dataset = []
    for _ in range(num_users):
        seq_len = random.randint(5, max_len)
        user_seq = [random.randint(1, num_items - 1) for _ in range(seq_len)]
        target = user_seq[-1]
        input_seq = user_seq[:-1]
        dataset.append((input_seq, target))
    return dataset

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        pad_len = self.max_len - len(input_seq)
        input_seq = [0] * pad_len + input_seq[-self.max_len:]
        return torch.tensor(input_seq), torch.tensor(target)

# Generate dummy dataset
train_data = generate_dummy_dataset(num_users=800)
test_data = generate_dummy_dataset(num_users=200)

train_loader = DataLoader(DummyDataset(train_data, max_seq_len), batch_size=64, shuffle=True)
test_loader = DataLoader(DummyDataset(test_data, max_seq_len), batch_size=64, shuffle=False)

# Initialize model
model = SASRec(num_users=num_users, num_items=num_items, max_seq_len=max_seq_len, hidden_units=64, num_heads=2, num_blocks=2, dropout_rate=0.2)
model = model.to(device)

# Train
train(model, train_loader, device, epochs=5)

# Evaluate
hit, ndcg = evaluate(model, test_loader, device)
print(f"Hit@10: {hit:.4f}, NDCG@10: {ndcg:.4f}")
