import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description="explainer")
    parser.add_argument("--model", type=str, default="LightGCN", help="Model name")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--embedding_dim", type=int, default=64, help="GNN Embedding dimension")
    return parser.parse_args()

args = parse_configure()
