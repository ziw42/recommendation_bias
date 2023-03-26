import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from tqdm import tqdm
import copy

if __name__ == "__main__":
    ### alpha = 1
    ### beta = 0.4
    parser = argparse.ArgumentParser(description='popularity compensation')
    parser.add_argument('--data', type=str, default='ML1M', help='path to eval in the downloaded folder')
    parser.add_argument('--method', type=str, default='MF', help='method to be evaluated')
    parser.add_argument('--alpha', type=float, default=0.4, help='alpha')
    parser.add_argument('--beta', type=float, default=1.2, help='beta')
    parser.add_argument('--split', type=int, default=1, help='split')

    args = parser.parse_args()

    print(args)

    with open('../Data/' + args.data + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        args.num_user = info['num_user']
        args.num_item = info['num_item']

    P = np.load('../Data/' + args.data + '/P_' + args.method + '.npy')
    Q = np.load('../Data/' + args.data + '/Q_' + args.method + '.npy')

    train_pop = np.load('../Data/' + args.data + '/train_pop.npy')
    # vali_pop = np.load('../Data/' + args.data + '/vali_pop.npy')
    test_pop = np.load('../Data/' + args.data + '/test_pop.npy')

    train_df = pd.read_csv('../Data/' + args.data + '/train_df.csv')

    print('sparsity = ' + str(len(train_df) / (args.num_item * args.num_user)))

    train_like = list(np.load('../Data/' + args.data + '/user_train_like.npy', allow_pickle=True))
    test_like = list(np.load('../Data/' + args.data + '/user_test_like.npy', allow_pickle=True))
    # vali_like = list(np.load('../Data/' + args.data + '/user_vali_like.npy', allow_pickle=True))

    Rec_list = []
    batch_num_user = int(args.num_user / args.split)
    for i in range(args.split):
        u_start = i * batch_num_user
        u_end = min((i + 1) * batch_num_user, args.num_user)
        P_batch = P[u_start:u_end, :]
        Rec = np.matmul(P_batch, Q.T)
        Rec_copy = copy.copy(Rec)
        user_score_norm = np.zeros((u_end - u_start, 1))
        for u in range(u_end - u_start):
            Rec_copy[u, train_like[u]] = 0
            user_score_norm[u, 0] = np.sqrt(np.sum(np.square(Rec_copy[u, :])) / (args.num_item - len(train_like[u])))
        rep_pop = 1. / (train_pop + 1)

        normed_Rec = Rec / user_score_norm
        pop_bias = np.matmul(user_score_norm, rep_pop.reshape(1, -1)) * (Rec / user_score_norm * args.beta + 1 - args.beta)

        pop_bias_norm = np.zeros((u_end - u_start, 1))
        for u in range(u_end - u_start):
            pop_bias[u, train_like[u]] = 0
            pop_bias_norm[u, 0] = np.sqrt(np.sum(np.square(pop_bias[u, :])) / (args.num_item - len(train_like[u])))

        norm_weight = user_score_norm / pop_bias_norm
        Rec = Rec + pop_bias * norm_weight * args.alpha
        Rec_list.append(Rec)

    Rec = np.concatenate(Rec_list, axis=0)
    utility.MP_test_model_withbias(Rec, test_like, train_like, train_pop, test_pop, n_workers=10, k=1)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!bias analysis!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    save_file = '../Data/' + args.data + '/PC_' + args.method + '.pkl'
    utility.ranking_analysis_save(Rec, train_like=train_like, test_like=test_like, train_pop=train_pop, test_pop=test_pop, save_file=save_file)
    utility.ranking_analysis_load(train_pop=train_pop, test_pop=test_pop, load_file=save_file)
    np.save('../Data/' + args.data + '/Rec_PC_' + args.method + '.npy', Rec)
    print()



