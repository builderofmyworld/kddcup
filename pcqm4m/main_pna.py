import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import degree
from tqdm import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import ModuleList
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch.nn import Sequential, ReLU, Linear
from models.pytorch_geometric.pna import PNAConvSimple
import torch.nn.functional as F

#from conv import PNANet

reg_criterion = torch.nn.L1Loss()


def my_collate(batch):
    return batch
    # pass

def train(model,device,train_loader,optimizer):
    model.train()

    total_loss = 0
    for step, data in enumerate(tqdm(train_loader, desc="Iteration")):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

def eval(model,device,loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred

def main_pan():
    #training settings
    parser = argparse.ArgumentParser(description='Principal neighbourhood aggregation on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any(default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='embedding dimensionality (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--radius', type=int, default=2,
                  help='radius (default: 2)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')
    #parser.add_argument('--node_embedding', type=torch.nn.modules.sparse.Embedding, default='Embedding(21, 75)', help='')
    #parser.add_argument('--edge_embedding', type=torch.nn.modules.sparse.Embedding, default='Embedding(4, 50)', help='')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    dataset = PygPCQM4MDataset(root='~/lsc/pcqm4m/dataset')
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()
    #transform = transforms.Compose([transforms.ToTensor()])

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    if args.save_test_dir != '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset[split_idx["train"]][0:10000]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        # print(d)
        deg += torch.bincount(d, minlength=deg.numel())

    class PNANet(torch.nn.Module):
        def __init__(self):
            super(PNANet, self).__init__()

            self.node_emb = AtomEncoder(emb_dim=70)

            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']

            self.convs = ModuleList()
            self.batch_norms = ModuleList()
            for _ in range(4):
                conv = PNAConvSimple(in_channels=70, out_channels=70, aggregators=aggregators,
                                     scalers=scalers, deg=deg, post_layers=1)
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(70))

            self.mlp = Sequential(Linear(70, 35), ReLU(), Linear(35, 17), ReLU(), Linear(17, 1))

        def forward(self, x, edge_index, edge_attr, batch):
            x = self.node_emb(x)

            for conv, batch_norm in zip(self.convs, self.batch_norms):
                h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
                x = h + x  # residual#
                x = F.dropout(x, 0.3, training=self.training)

            x = global_mean_pool(x, batch)
            return self.mlp(x)

    model = PNANet().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        print(type(train_loader))
        train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            #if args.save_test_dir != '':
            #    print('Predicting on test data...')
            #    y_pred = test(model, device, test_loader)
            #    print('Saving test submission file...')
            #    evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main_pan()