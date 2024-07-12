"""
Inference of a diffusion model for recommendation
"""

import argparse
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

import models.gaussian_diffusion as gd
import evaluate_utils
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ML-1M', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/ML-1M/', help='load data path')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--log_name', type=str, default='log', help='the log name')

parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
parser.add_argument('--n_hops', type=int, default=2, help='Number of multi-hop neighbors')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()

args.data_path = args.data_path
if args.dataset == 'ML-1M':
    args.steps = 5
    args.noise_scale = 0.01
    args.noise_min = 0.001
    args.noise_max = 0.01
    args.n_hops = 3
elif args.dataset == 'anime':
    args.steps = 10
    args.noise_scale = 0.003
    args.noise_min = 0.0001
    args.noise_max = 0.01
    args.n_hops = 2
elif args.dataset == 'yelp2018':
    args.steps = 20
    args.noise_scale = 0.01
    args.noise_min = 0.001
    args.noise_max = 0.01
    args.n_hops = 2

else:
    raise ValueError

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATA LOAD ###
train_path = args.data_path + 'train_list_' + args.dataset + '.npy'
valid_path = args.data_path + 'valid_list_' + args.dataset + '.npy'
test_path = args.data_path + 'test_list_' + args.dataset + '.npy'

print("{}-hop neighbors are taken into account".format(args.n_hops))
if args.n_hops == 2:
    sec_hop = torch.load(args.data_path + 'two_hop_rates_items_' + args.dataset + '.pt')
    multi_hop = sec_hop
elif args.n_hops == 3:
    multi_hop = torch.load(args.data_path + 'multi_hop_inters_' + args.dataset + '.pt')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion2(torch.FloatTensor(train_data.A), multi_hop)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size)

mask_tv = train_data + valid_y_data

print('data ready.')


### CREATE DIFFUISON ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device)
diffusion.to(device)

### CREATE DNN ###
model_path = "saved_models/"
if args.dataset == "anime":
    model_name = "CAM_2hops_anime.pth"
elif args.dataset == "yelp2018":
    model_name = "CAM_2hops_yelp2018.pth"
elif args.dataset == "ML-1M":
    model_name = "CAM_3hops.pth"


def evaluate(data_loader, data_te, mask_his, topN, model):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, (batch, batch_2) in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            batch_2 = batch_2.to(device)
            prediction = diffusion.p_sample(model, batch, batch_2, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results


model = torch.load(model_path + model_name).to(device)  # batch=50

print("Initial models ready.")

valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN), model)
test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN), model)
evaluate_utils.print_results(None, valid_results, test_results)
