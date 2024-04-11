"""
This is the file from the authors github accuount
"""
import matplotlib.pyplot as plt
import os
import time
import torch
import pickle
import argparse
from datasets import Traindataset
from torch.utils.data import DataLoader
import scipy.sparse as sp
from AGRAN import AGRAN
from tqdm import tqdm
from utils_ag import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(42)

'''
                Foursquare    Gowalla
coefficient 𝜇       1𝑒−4       1𝑒−3         V
learning rate       1𝑒−3       1𝑒−3         V
dropout rate        0.3        0.2          V
# GCN layer         4           2           - in text (model_ag.py)
max len of seq      50,                     V
dist, time thresh   256                     V,V
regularization 𝜆     1                      V
# anchor             500                    - only in anchor version. N/A here

EPOCH best results  50-60
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='four-sin')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=3, type=int)
parser.add_argument('--num_epochs', default=121, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--dis_span', default=256, type=int)
parser.add_argument('--kl_reg', default=1.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)


def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj


if __name__ == '__main__':
    dataset = data_partition('datasets/foursquare_given/four-sin.txt')
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    num_batch = len(user_train) // args.batch_size

    tra_adj_matrix = sp.load_npz('datasets/foursquare_given/sin_transaction_kl_notest.npz')
    prior = torch.FloatTensor(tra_adj_matrix.todense()).to(args.device)

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print(itemnum)
    print(usernum)

    # delete older files
    try:
        os.remove(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'))
        os.remove(os.path.join(args.dataset + '_' + args.train_dir, 'eval_results.txt'))
    except Exception as e:
        pass

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'a')
    f2 = open(os.path.join(args.dataset + '_' + args.train_dir, 'eval_results.txt'), 'a')
    f.write('\t'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]) + '\n')
    f.flush()

    try:
        relation_matrix = pickle.load(
            open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
    except:
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(relation_matrix,
                    open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))

    try:
        dis_relation_matrix = pickle.load(
            open('data/relation_dis_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.dis_span), 'rb'))
    except:
        dis_relation_matrix = Relation_dis(user_train, usernum, args.maxlen, args.dis_span)
        pickle.dump(dis_relation_matrix,
                    open('data/relation_dis_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.dis_span), 'wb'))

    train_dataset = Traindataset(user_train, relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    model = AGRAN(usernum, itemnum, args.device, args.maxlen, args.dis_span, args.hidden_units, args.dropout_rate, args.time_span).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    ce_criterion = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    T = 0.0
    t0 = time.time()

    all_NDCG2 = []
    all_NDCG5 = []
    all_NDCG10 = []
    all_recall2 = []
    all_recall5 = []
    all_recall10 = []
    all_loss = []

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only: break
        tmp_loss = []
        for step, instance in tqdm(enumerate(dataloader), total=num_batch, ncols=70, leave=False, unit='b'):
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
            kl_reg = args.kl_reg
            if epoch >= 0:
                loss += kl_reg * a

            tmp_loss.append(loss.item())

            loss.backward()
            adam_optimizer.step()

        all_loss.append(sum(tmp_loss)/len(tmp_loss))

        if epoch % 2 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            ### for validation ###
            NDCG, HR = evaluate_vaild(model, dataset, args)

            ### for test ###
            NDCG, HR = evaluate_test(model, dataset, args)

            print('epoch:%d, time: %f(s), NDCG (@2: %.4f, @5: %.4f,@10: %.4f), Recall (@2: %.4f, @5: %.4f,@10: %.4f)'
                  % (epoch, T, NDCG[0], NDCG[1], NDCG[2], HR[0], HR[1], HR[2]))
            f2.write('epoch:%d, time: %f(s), NDCG (@2: %.4f, @5: %.4f,@10: %.4f), Recall (@2: %.4f, @5: %.4f,@10: %.4f)'
                     % (epoch, T, NDCG[0], NDCG[1], NDCG[2], HR[0], HR[1], HR[2]))
            f2.flush()
            f.write('epoch:' + str(epoch) + ' ' + str(float('%.4f' % NDCG[2].item())) + ' ' + str(
                float('%.4f' % HR[2])) + '\n')
            f.flush()

            all_NDCG2.append(NDCG[0])
            all_NDCG5.append(NDCG[1])
            all_NDCG10.append(NDCG[2])
            all_recall2.append(HR[0])
            all_recall5.append(HR[1])
            all_recall10.append(HR[2])

            t0 = time.time()

    f.close()
    f2.close()
    print("Done")

    # plot separately
    x_ticks = [x for x in range(1, args.num_epochs + 1) ]
    plt.plot(all_loss, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('img/our/loss.png', bbox_inches='tight')
    plt.clf()

    all_plots = [all_NDCG2, all_NDCG5, all_NDCG10, all_recall2, all_recall5, all_recall10]
    all_labels = ['NDCG@2','NDCG@5','NDCG@10','HR@2','HR@5','HR@10']
    x_ticks = [x for x in range(1, args.num_epochs+1) if x % 2 == 0]
    for i in range(len(all_plots)):
        plt.plot(x_ticks, all_plots[i], label=all_labels[i])
        plt.xlabel('epoch')
        plt.ylabel(all_labels[i].split('@')[0])
        plt.legend()
        plt.savefig('img/our/'+all_labels[i]+'.png', bbox_inches='tight')
        plt.clf()

    # plot all togather
    for i in range(len(all_plots)):
        if i == 3:
            plt.xlabel('epoch')
            plt.ylabel(all_labels[i].split('@')[0])
            plt.legend()
            plt.savefig('img/our/combined_' + all_labels[i].split('@')[0] + '.png', bbox_inches='tight')
            plt.clf()

        plt.plot(x_ticks, all_plots[i], label=all_labels[i])

        if i == len(all_labels)-1:
            plt.xlabel('epoch')
            plt.ylabel(all_labels[i].split('@')[0])
            plt.legend()
            plt.savefig('img/our/combined_' + all_labels[i].split('@')[0] + '.png', bbox_inches='tight')
            plt.clf()


