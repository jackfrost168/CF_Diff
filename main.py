"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
#from models.MultiheadAttentionModel import MultiheadAttentionModel
#from models.MultiheadAttentionModel_2 import MultiheadAttentionModel
#from models.MultiheadAttentionModel_2_ablation_study import MultiheadAttentionModel
#from models.MultiheadAttentionModel_2_ablation_study_no_encoder import MultiheadAttentionModel
#from models.MultiheadAttentionModel_2_concat_self_atten import MultiheadAttentionModel
#from models.MultiheadAttentionModel_2_1st_hop import MultiheadAttentionModel
from models.MultiheadAttentionModel_2_multihop import MultiheadAttentionModel

import evaluate_utils
import data_utils
from copy import deepcopy
from torchsummary import summary
import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/Yelp/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[256,256]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.001, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.001, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

print("torch version:", torch.__version__)

args = parser.parse_args()
print("args:", args)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#device = torch.device("cuda:0" if args.cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
data_name = 'Yelp'
train_path = args.data_path + 'train_list_' + data_name + '.npy'
valid_path = args.data_path + 'valid_list_' + data_name + '.npy'
test_path = args.data_path + 'test_list_' + data_name + '.npy'

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
print(train_data.shape)

#print(train_data[0])
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

sec_hop = torch.load(args.data_path + 'multi_hop_rates_items_Yelp.pt')
train_loader_sec_hop = DataLoader(sec_hop, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0, worker_init_fn=worker_init_fn)
test_loader_sec_hop = DataLoader(sec_hop, batch_size=args.batch_size, shuffle=False)

# train_loader_sec_hop = train_loader
# test_loader_sec_hop = test_loader
# print("train_dataset", train_dataset[100][0:40])
# print("Sec hop:", sec_hop[100][0:40])

print("args.tst_w_val:", args.tst_w_val)
if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
print("output_dim:", out_dims)
in_dims = out_dims[::-1]
print(in_dims)
#model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
#model = MultiheadAttentionModel(n_item, 1, 2, args.emb_size).to(device)
model = MultiheadAttentionModel(64, 1, 2, in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
#print(summary(model, (100,)))
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def evaluate(data_loader, data_loader_sec_hop, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        #for batch_idx, batch in enumerate(data_loader):
        for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(data_loader), enumerate(data_loader_sec_hop)):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            batch_2 = batch_2.to(device)
            prediction = diffusion.p_sample(model, batch, batch_2, args.sampling_steps, args.sampling_noise)
            #print(prediction)
            prediction[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

if __name__ == '__main__':
    # for (batch_idx_1, batch_1), (batch_idx_2, batch_2) in zip(enumerate(train_loader), enumerate(train_loader_sec_hop)):
    #     print("1:", batch_1[100][0:30])
    #     print("2:", batch_2[100][0:30])
    # for batch_idx_1, batch_1 in enumerate(train_loader):
    #     print("1:", batch_idx_1, batch_1[17][0:30])
    #
    # for batch_idx_2, batch_2 in enumerate(train_loader_sec_hop):
    #     print("2:", batch_idx_2, batch_2[17][0:30])


    best_recall, best_epoch = -100, 0
    best_test_result = None
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= 20:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0

        for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(train_loader), enumerate(train_loader_sec_hop)):
            #print(batch_idx)
            batch = batch.to(device)
            batch_2 = batch_2.to(device)
            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, batch_2, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            valid_results = evaluate(test_loader, test_loader_sec_hop, valid_y_data, train_data, eval(args.topN))
            if args.tst_w_val:
                test_results = evaluate(test_twv_loader, test_loader_sec_hop, test_y_data, mask_tv, eval(args.topN))
            else:
                test_results = evaluate(test_loader, test_loader_sec_hop, test_y_data, mask_tv, eval(args.topN))
            evaluate_utils.print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall: # recall@20 as selection
                best_recall, best_epoch = valid_results[1][1], epoch
                best_results = valid_results
                best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                    .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                    args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))

        print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)

    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, best_results, best_test_results)
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





