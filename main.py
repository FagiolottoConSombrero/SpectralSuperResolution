from models.mine import *
from engine import *
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.nn import MSELoss, SmoothL1Loss

parser = argparse.ArgumentParser(description='Single Image Super Resolution')
parser.add_argument('--model', type=str, default='1', help='model id')
parser.add_argument('--t_data_path', type=str, default='', help='Train Dataset path')
parser.add_argument('--v_data_path', type=str, default='', help='Val Dataset path')
parser.add_argument('--batch_size', type=int, default='2', help='Training batch size')
parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")
parser.add_argument('--save_path', type=str, default='', help="Path to model checkpoint")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")


def main():
    opt = parser.parse_args()
    print(opt)

    print("===> Loading data")
    train_set = AradDataset(opt.t_data_path)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)

    valid_set = AradDataset(opt.v_data_path, train=False)
    valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batch_size, shuffle=True)

    print("===> Building model")
    if opt.model == '1':
        model = SPAN(31, 31)

    model = model.to(opt.device)
    loss = SmoothL1Loss()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    print("===> Starting Training")
    train(train_loader,
            valid_loader,
            model,
            opt.epochs,
            optimizer,
            opt.device,
            opt.save_path,
            loss,
            opt.lr)


if __name__ == "__main__":
    main()