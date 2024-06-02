import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description="explainer")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="finetune", help="finetune or generate")
    return parser.parse_args()

args = parse_configure()
