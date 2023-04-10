from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import PygNodePropPredDataset

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
print('Loading OGB arxiv dataset...')
arxiv_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
edge_list = arxiv_dataset[0].edge_index
split_idx = arxiv_dataset.get_idx_split()
idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
idx_train = torch.squeeze(idx_train)
idx_val = torch.squeeze(idx_val)
idx_test = torch.squeeze(idx_test)
features = arxiv_dataset[0].x
labels = arxiv_dataset[0].y
labels = torch.squeeze(labels)
edge_list = arxiv_dataset[0].edge_index
num_nodes = arxiv_dataset[0].num_nodes
# adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
# for i in range(edge_list.shape[1]):
#     adj[edge_list[0][i]][edge_list[1][i]] = 1
#     adj[edge_list[1][i]][edge_list[0][i]] = 1

# 创建一个大小为 [num_nodes, num_nodes] 的稀疏邻接矩阵
indices = torch.cat((edge_list, torch.flip(edge_list, [0])), 1)  # 反转边列表并连接，以考虑无向图的双向边
values = torch.ones(indices.size(1))  # 值向量，其长度与边的数量相同
adj_matrix_sparse = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))


# 计算节点的度向量
degree = torch.sparse.sum(adj_matrix_sparse, dim=1).to_dense()

# 计算度矩阵的平方根的逆（D^(-1/2)）
degree_inv_sqrt = degree.pow(-1/2)
degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # 处理零度节点

# 构建对角矩阵 D^(-1/2)
diag_indices = torch.arange(len(degree_inv_sqrt), dtype=torch.long)
diag_degree_inv_sqrt = torch.sparse.FloatTensor(
    torch.stack([diag_indices, diag_indices]),
    degree_inv_sqrt,
    (len(degree_inv_sqrt), len(degree_inv_sqrt))
)

# 计算归一化的邻接矩阵：D^(-1/2) * A * D^(-1/2)
normalized_adj_matrix = torch.sparse.mm(
    torch.sparse.mm(diag_degree_inv_sqrt, adj_matrix_sparse),
    diag_degree_inv_sqrt
)

adj = normalized_adj_matrix
print('Done loading OGB arxiv dataset...')

# Model and optimizer
print('Creating model...')
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

print('Done creating model...')

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
