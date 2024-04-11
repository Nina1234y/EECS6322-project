import os
import time
import torch
import argparse
from datasets import Traindataset
from torch.utils.data import DataLoader
from AGRAN import AGRAN
from tqdm import tqdm
from preprocess import preprocess_and_load_dataset
import scipy
import utils
import numpy as np
from utils_ag import *

# -----------------------------
# Parameters
# -----------------------------
# hard set parameters based on the paper for Foursquare
mu_coeff = 0.0001
lr = 0.001
dropout_rate = 0.3
GCN_layer = 4
max_seq_len = 50
dis_thresh = 256
time_thresh = 256
regularization = 1
num_epochs = 60  # based on a comment on github
batch_size = 128
hidden_units = 64
state_dict_path = None

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()

def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj

# -----------------------------
# Main method
# -----------------------------
# Preprocess Data
train, valid, test, num_users, num_pois, num_pois = preprocess_and_load_dataset()
batches = len(train) // batch_size

tra_adj_matrix = scipy.sparse.load_npz('datasets/foursquare_given/sin_transaction_kl_notest.npz')
prior = torch.FloatTensor(tra_adj_matrix.todense()).to(args.device)

avg_seq_len = sum([len(train[u]) for u in train])/len(train)
print("average sequance length: ", avg_seq_len)

# get temporal and distance relation matrices
# Spatial-Temporal Dependencies Encoding -> section 4.3.2
rmatrix = utils.get_transition_distribution(num_users, max_seq_len, train, time_thresh)
dmatrix = utils.get_distance_distribution(num_users, max_seq_len, train, dis_thresh)

# load data
train_dataset = Traindataset(train, rmatrix, dmatrix, num_pois, max_seq_len)
dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = AGRAN(num_users, num_pois, args.device, max_seq_len, dis_thresh, hidden_units, dropout_rate, time_thresh).to(args.device)

# -------- from their github ---------:
# Remove older files if they exist
try:
    os.remove('train/log.txt')
    os.remove('train/eval_results.txt')
except Exception as e:
    pass

# Open log files for writing
f = open('train/log.txt', 'a')
eval_file = open('train/eval_results.txt', 'a')

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_uniform_(param.data)
    except:
        pass

model.train()

epoch_start_idx = 1
if state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(args.device)))
        tail = state_dict_path[state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except:
        print('failed loading state_dicts, pls check file path: ', end="")
        print(state_dict_path)

ce_criterion = torch.nn.CrossEntropyLoss()
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
parameters = [{'params': weight_decay_list},
              {'params': no_decay_list, 'weight_decay': 0.}]

adam_optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.98), weight_decay=mu_coeff)

T = 0.0
t0 = time.time()
global_it = 0

for epoch in range(epoch_start_idx, num_epochs + 1):
    model.train()
    for step, instance in tqdm(enumerate(dataloader), total=batches, ncols=70, leave=False, unit='b'):
        u, seq, time_seq, pos, neg, time_matrix, dis_matrix = instance
        pos_logits, neg_logits, fin_logits, padding, support = model(u, seq, time_matrix, dis_matrix, pos, neg)
        a = kl_loss(torch.log(torch.softmax(mask(support), dim=-1) + 1e-9), torch.softmax(mask(prior), dim=-1))
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                               device=args.device)
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        pos_label_for_crosse = pos.numpy().reshape(-1)

        indices_for_crosse = np.where(pos_label_for_crosse != 0)

        pos_label_cross = torch.tensor(pos_label_for_crosse[indices_for_crosse], device=args.device)
        loss = ce_criterion(fin_logits[indices_for_crosse], pos_label_cross.long())
        kl_reg = regularization
        if epoch >= 0:
            loss += kl_reg * a

        loss.backward()
        adam_optimizer.step()


    if epoch % 2 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        ### for validation ###
        NDCG, HR = evaluate_vaild(model, (train, valid, test, num_users, num_pois, num_pois), max_seq_len, time_thresh, dis_thresh, batch_size)

        ### for test ###
        NDCG, HR = evaluate_test(model, (train, valid, test, num_users, num_pois, num_pois), max_seq_len, time_thresh, dis_thresh, batch_size)

        print('epoch:%d, time: %f(s), NDCG (@2: %.4f, @5: %.4f,@10: %.4f), Recall (@2: %.4f, @5: %.4f,@10: %.4f)'
              % (epoch, T, NDCG[0], NDCG[1], NDCG[2], HR[0], HR[1], HR[2]))
        eval_file.write('epoch:%d, time: %f(s), NDCG (@2: %.4f, @5: %.4f,@10: %.4f), Recall (@2: %.4f, @5: %.4f,@10: %.4f)'
                 % (epoch, T, NDCG[0], NDCG[1], NDCG[2], HR[0], HR[1], HR[2]))
        eval_file.flush()
        f.write(
            'epoch:' + str(epoch) + ' ' + str(float('%.4f' % NDCG[2].item())) + ' ' + str(float('%.4f' % HR[2])) + '\n')
        f.flush()
        t0 = time.time()
        model.train()

f.close()
eval_file.close()
print("Done")
