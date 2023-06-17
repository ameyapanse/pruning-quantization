from torch_sparse import SparseTensor
from typing import Tuple
import copy, time
import torch
from torch import Tensor
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr, Yelp, PPI, Reddit, GNNBenchmarkDataset
from torch_geometric.data import Batch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from lsp.tst.ogb.main_pyg_with_pruning import prune_dataset
import os

class MyToSparseTensor(object):
    r"""Converts the :obj:`edge_index` attribute of a data object into a
    (transposed) :class:`torch_sparse.SparseTensor` type with key
    :obj:`adj_.t`.

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object to a :obj:`SparseTensor` as late as possible, since
        there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`False`, will not
            fill the underlying :obj:`SparseTensor` cache.
            (default: :obj:`True`)
    """
    def __init__(self, remove_edge_index: bool = True,
                 fill_cache: bool = True):
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache

    def __call__(self, data):
        assert data.edge_index is not None

        (row, col), N, E = data.edge_index, data.num_nodes, data.num_edges
        perm = (col * N + row).argsort()
        row, col = row[perm], col[perm]

        if self.remove_edge_index:
            data.edge_index = None

        value = None
        for key in ['edge_weight', 'edge_attr', 'edge_type']:
            if data[key] is not None:
                value = data[key][perm]
                if self.remove_edge_index:
                    data[key] = None
                break

        for key, item in data:
            if item.size(0) == E:
                data[key] = item[perm]

        data.adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=None, is_sorted=True)

        if self.fill_cache:  # Pre-process some important attributes.
            data.adj_t.storage.rowptr()
            data.adj_t.storage.csr2csc()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'

def norm_adj(data, conv_type):
    if conv_type == 'GCN':
        data.adj_t = data.adj_t.set_diag() # self-loop
        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-1 / 2)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        data.adj_t = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)
    elif conv_type == 'SAGE':
        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        data.adj_t = deg_inv.view(-1, 1) * data.adj_t
    elif conv_type == 'GAT':
        data.adj_t = data.adj_t.set_diag() # self-loop
        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        data.adj_t = deg_inv.view(-1, 1) * data.adj_t
    else:
        raise ValueError('GNN conv type not supported')
    return data

def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

def prepare_batch_input(x, batch, device) :
    subset, edge_index, edge_w = batch[0]
    batch_idx = batch[-1]

    from torch_geometric.utils import add_remaining_self_loops

    new_edge_index = add_remaining_self_loops(edge_index)[0]
    num_newly_added_edges = new_edge_index.shape[1]-edge_index.shape[1]

    num_B = batch_idx.shape[0]
    dim = subset.shape[0]
    num_B_prime = dim - num_B

    print(f'num_B_prime:{num_B_prime}, new edges:{num_newly_added_edges}')

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_w, sparse_sizes=(dim, dim))
    batch_input = x[batch_idx].to(device), (batch_idx, subset, adj.to(device))
    return batch_input, (num_B, num_B_prime)

def prepare_batch_input_link(x, batch, device) :
    subset, edge_index, edge_w = batch[0]
    batch_idx = batch[-1]

    num_B = batch_idx.shape[0]
    dim = subset.shape[0]
    num_B_prime = dim - num_B

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_w, sparse_sizes=(dim, dim))
    batch_input = x[batch_idx].to(device), (batch_idx, subset, adj.to(device))

    edge_mask = (edge_index[0] < num_B) &  (edge_index[1] < num_B)
    src, dst = edge_index[:, edge_mask]

    return batch_input, (num_B, num_B_prime), (src, dst)

def metis(adj_t: SparseTensor, num_parts: int, recursive: bool = False,
          log: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Computes the METIS partition of a given sparse adjacency matrix
    :obj:`adj_t`, returning its "clustered" permutation :obj:`perm` and
    corresponding cluster slices :obj:`ptr`."""
    if log:
        t = time.perf_counter()
        print(f'Computing METIS partitioning with {num_parts} parts...',
              end=' ', flush=True)
    num_nodes = adj_t.size(0)
    if num_parts <= 1:
        perm, ptr = torch.arange(num_nodes), torch.tensor([0, num_nodes])
    else:
        rowptr, col, _ = adj_t.csr()
        cluster = torch.ops.torch_sparse.partition(rowptr, col, None, num_parts, recursive)
        cluster, perm = cluster.sort()
        ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)
    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    return perm, ptr

def permute(data: Data, perm: Tensor, log: bool = True) -> Data:
    r"""Permutes a :obj:`data` object according to a given permutation
    :obj:`perm`."""
    if log:
        t = time.perf_counter()
        print('Permuting data...', end=' ', flush=True)
    data = copy.copy(data)
    for key, value in data:
        if isinstance(value, Tensor) and value.size(0) == data.num_nodes:
            data[key] = value[perm]
        elif isinstance(value, Tensor) and value.size(0) == data.num_edges:
            raise NotImplementedError
        elif isinstance(value, SparseTensor):
            data[key] = value.permute(perm)
    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    return data


def inductive_data(dataset):
    data = Batch.from_data_list(dataset)
    data.batch, data.ptr = None, None
    data['train_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
    return data

def index2mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def get_data(args, lsp_args=None, device=None) :
    if args.dataset in {'arxiv', 'products'}:
        dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}',
                                         transform=T.ToSparseTensor(),
                                         root=os.path.join(args.data_root, 'ogb'))
    elif args.dataset == 'flickr':
        dataset = Flickr(root=os.path.join(args.data_root, 'graph', args.dataset),
                         transform=MToSparseTensor())
    elif args.dataset == 'yelp':
        dataset = Yelp(root=os.path.join(args.data_root, 'graph', args.dataset),
                       transform=T.ToSparseTensor())
    elif args.dataset == 'reddit':
        dataset = Reddit(root=os.path.join(args.data_root, 'graph', args.dataset),
                         transform=T.ToSparseTensor())
    elif args.dataset == 'ppi':
        print('PPI loaded')
        # tr_x, tr_y, tr_e, v_x, v_y, v_e, tst_x, tst_y, tst_e = load_graph_data(lsp_args, device=device)
        # dataset = GraphDataset(tr_x, tr_y, tr_e, transform=T.ToSparseTensor())
        # val_dataset = GraphDataset(v_x, v_y, v_e, transform=T.ToSparseTensor())
        # test_dataset = GraphDataset(tst_x, tst_y, tst_e, transform=T.ToSparseTensor())
        #
        dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                      transform=MyToSparseTensor(), split='train')
        # dataset2 = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
        #               transform=T.ToSparseTensor(), split='train')
        val_dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                          transform=MyToSparseTensor(), split='val')
        test_dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                           transform=MyToSparseTensor(), split='test')

        if lsp_args is not None:
            train = [dataset.data]
            val = [val_dataset.data]
            test = [test_dataset.data]
            pruning_params, prunning_ratio = prune_dataset(train, lsp_args, pruning_params=None)
            prune_dataset(val, lsp_args, pruning_params=None)
            prune_dataset(test, lsp_args, pruning_params=None)
            dataset.data.edge_attr = None
            val_dataset.data.edge_attr = None
            test_dataset.data.edge_attr = None
            print(pruning_params, prunning_ratio)

        data, val_data, test_data = inductive_data(dataset), inductive_data(val_dataset), inductive_data(
            test_dataset)
    elif args.dataset == 'cluster':
        print('CLUSTER loaded')
        kwargs = {'root': os.path.join(args.data_root, 'graph', args.dataset), 'name': 'CLUSTER',
                  'transform': T.ToSparseTensor()}
        dataset = GNNBenchmarkDataset(split='train', **kwargs)
        val_dataset = GNNBenchmarkDataset(split='val', **kwargs)
        test_dataset = GNNBenchmarkDataset(split='test', **kwargs)
        data, val_data, test_data = inductive_data(dataset), inductive_data(val_dataset), inductive_data(
            test_dataset)
    else:
        raise ValueError('Dataset not supported!')

    evaluator, cluster_indices = None, None

    # transductive datasets
    if args.dataset not in {'ppi', 'cluster'} :
        data, val_data, test_data = dataset[0], None, None
        data.adj_t = data.adj_t.to_symmetric()

        if args.dataset in {'arxiv', 'products'}:
            split_idx = dataset.get_idx_split()
            data.train_mask = index2mask(split_idx['train'], data.num_nodes)
            data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index2mask(split_idx['test'], data.num_nodes)
            evaluator = Evaluator(name=f'ogbn-{args.dataset}')

        if args.sampler_type == 'cluster' :
            num_parts = args.num_parts
            perm, ptr = metis(data.adj_t, num_parts=num_parts, log=True)
            data = permute(data, perm, log=True)
            n_id = torch.arange(data.num_nodes)
            cluster_indices = n_id.split((ptr[1:] - ptr[:-1]).tolist())

        data = norm_adj(data, args.conv_type)

    # inductive datasets
    else:
        if args.sampler_type == 'cluster':
            raise NotImplementedError

        data.adj_t, val_data.adj_t, test_data.adj_t = data.adj_t.to_symmetric(), val_data.adj_t.to_symmetric(), test_data.adj_t.to_symmetric()
        data, val_data, test_data = norm_adj(data, args.conv_type), norm_adj(val_data, args.conv_type), norm_adj(
            test_data, args.conv_type)

    if args.split :
        if data.num_features % args.num_D != 0 :
            padding_dim = args.num_D - data.num_features % args.num_D
            data.x = torch.cat([data.x, torch.zeros((data.num_nodes, padding_dim))], dim=-1)

            if args.dataset in {'ppi', 'cluster'} :
                val_data.x = torch.cat([val_data.x, torch.zeros((val_data.x.shape[0], padding_dim))], dim=-1)
                test_data.x = torch.cat([test_data.x, torch.zeros((test_data.x.shape[0], padding_dim))], dim=-1)

        if args.hidden_channels % args.num_D != 0 :
            raise ValueError('Cannot fully split hidden features')

    return data, val_data, test_data, dataset, evaluator, cluster_indices



