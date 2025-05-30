# sasrec/baseline/demo.py
import torch
from model import SASRec
from trainer import train
from torch.utils.data import DataLoader, TensorDataset
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    return parser.parse_args()

def main():
    args = get_args()
    model = SASRec(user_num=100, item_num=500, args=args)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # random data test
    dummy_seq = torch.randint(1, 500, (1000, args.max_seq_len))
    dummy_label = torch.randint(1, 500, (1000,))
    dataset = TensorDataset(dummy_seq, dummy_label)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    avg_loss = train(model, dataloader, optimizer, device=model.device)
    print(f"Training loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
