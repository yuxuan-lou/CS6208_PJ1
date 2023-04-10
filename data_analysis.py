from ogb.nodeproppred import PygNodePropPredDataset
from pygcn.utils import load_data

arxiv_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
split_idx = arxiv_dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

#Data(num_nodes=169343, edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])
graph = arxiv_dataset[0]

print(arxiv_dataset[0]['edge_index'].shape)

cora_dataset = load_data(path="./data/cora/", dataset="cora")


#adj
print(cora_dataset[0].shape)

#features
print(cora_dataset[1].shape)

#labels
print(cora_dataset[2].shape)

#idx_train
print(cora_dataset[3].shape)

#idx_val
print(cora_dataset[4].shape)

#idx_test
print(cora_dataset[5].shape)

