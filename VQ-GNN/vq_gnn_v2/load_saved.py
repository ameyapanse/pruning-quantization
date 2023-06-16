import torch
from models import LowRankGNN
from utils.parser import parse
from utils.misc import compute_micro_f1, prepare_batch_input, get_data

args = parse()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
data, val_data, test_data, dataset, evaluator, cluster_indices = get_data(args)

num_N = data.num_nodes
if args.batch_size <= 0:
    args.batch_size = num_N
if args.test_batch_size <= 0 :
    args.test_batch_size = num_N

model = LowRankGNN(data.num_features, args.hidden_channels, dataset.num_classes,
                       args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                       args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                       args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                       args.dropbranch, args.skip, args.use_gcn, args.commitment_cost,
                       args.grad_scale, args.act, args.weight_ahead, args.bn_flag,
                       args.warm_up, args.momentum, args.conv_type, args.transformer_flag,
                       args.alpha_dropout_flag).to(device)

model.load_state_dict(torch.load('models/vq_gat.pth'))
model.eval()