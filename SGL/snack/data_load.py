'''
coding: utf-8
started on 2022/06/13
end on 2022/xx/xx
@author zelo2
'''

import numpy as np
import random
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size, args):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # basic parameters
        self.n_user, self.n_item = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        # contrastive learning parameter
        self.ssl_ratio = eval(args.ssl_ratio)[1]  # 1-amazon-book
        self.aug_type = args.aug_type

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    user_id = int(l[0])  # user id
                    item_id_sequence = [int(i) for i in l[1:]]  # interacted item sequence

                    self.exist_users.append(user_id)
                    self.n_train += len(item_id_sequence)

                    self.n_user = max(self.n_user, user_id)
                    self.n_item = max(self.n_item, max(item_id_sequence))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    '''存在只有user_id没有item_id的test data'''
                    try:
                        item_id_sequence = [int(i) for i in l.split(' ')[1:]]  # interacted item sequence
                    except Exception:
                        continue

                    self.n_item = max(self.n_item, max(item_id_sequence))
                    self.n_test += len(item_id_sequence)

        self.n_item += 1  # start from 0
        self.n_user += 1

        # Construct the R matrix.
        # Since the high sparsity of og dataset, we use scipy.sparse.dok_matrix for R construction.
        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.train_set, self.test_set = {}, {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    data = [int(i) for i in l.split(' ')]
                    train_user_id = data[0]
                    train_item_id_sequence = data[1:]

                    for i in train_item_id_sequence:
                        self.R[train_user_id, i] = 1

                    self.train_set[train_user_id] = train_item_id_sequence

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')

                    try:
                        data = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    test_user_id = data[0]
                    test_item_id_sequence = data[1:]

                    self.test_set[test_user_id] = test_item_id_sequence

    def creat_adj_mat(self):
        adj_matrix = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        adj_matrix = adj_matrix.tolil()
        R = self.R.tolil()  # dok_matrix -> lil_matrix

        # "A" matrix in the paper
        adj_matrix[:self.n_user, self.n_user:] = R
        adj_matrix[self.n_user:, :self.n_user] = R.T
        adj_matrix = adj_matrix.todok()  # lil_matrix -> dok_matrix

        def mean_adj_single(adj):
            # D^-1 * A
            row_sum = np.array(adj.sum(1))  # sum each row (M + N)
            d_1 = np.power(row_sum, -1).flatten()  # -1次方
            # np.flatten():该函数返回一个折叠成一维的数组
            d_1[np.isinf(d_1)] = 0.  # 0/0=inf
            d_1_matrix = sp.diags(d_1)  # diagonal matrix

            result = d_1_matrix.dot(adj)  # [M+N,M+N] dot [M+N, M+N]
            return result.tocoo()  # dok_matrix -> coo_matrix

        def norm_adj_single(adj):
            # D^-(1/2) * A * D^-(1/2)
            row_sum = np.array(adj.sum(1))  # sum each row
            d_half = np.power(row_sum, -0.5).flatten()
            d_half[np.isinf(d_half)] = 0.
            d_half_matrix = sp.diags(d_half)

            result = d_half_matrix.dot(adj).dot(d_half_matrix)
            return result.tocoo()


        norm_adj_matrix = norm_adj_single(adj_matrix)

        norm_adj_matrix_plus_I = norm_adj_single(adj_matrix) + sp.eye(adj_matrix.shape[0])

        return norm_adj_matrix.tocsr(), norm_adj_matrix_plus_I.tocsr()

    def creat_aug_adj_matrix_(self, aug_type):
        '''
        :param aug_type: 'ND'-Node Drop 'ED'-Edge Drop
        :return: augmentation sub-graphs
        '''

        adj_matrix = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        adj_matrix = adj_matrix.tolil()

        R_hat = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)


        if aug_type == 'ND':  # node dropout
            user_drop_num = int(self.n_user * self.ssl_ratio)
            item_drop_num = int(self.n_item * self.ssl_ratio)
            drop_user_idx = random.sample(range(self.n_user), user_drop_num)
            drop_item_idx = random.sample(range(self.n_item), item_drop_num)

            user_drop_indicator = np.ones(self.n_user)
            item_drop_indicator = np.ones(self.n_item)
            user_drop_indicator[drop_user_idx] = 0
            item_drop_indicator[drop_item_idx] = 0
            diag_user_drop = sp.diags(user_drop_indicator)
            diag_item_drop = sp.diags(item_drop_indicator)

            # Construct R hat denoting the rating matrix after node drop
            R = self.R.tolil()  # rating matrix
            R_hat = diag_user_drop.dot(R).dot(diag_item_drop)
        elif aug_type == 'ED':  # edge dropout
            drop_edge_num = int(self.n_train * self.ssl_ratio)
            user_idx = self.R.tocoo().row
            item_idx = self.R.tocoo().col
            sample_drop_idx = random.sample(range(len(user_idx)), drop_edge_num)
            for drop_edge_idx in sample_drop_idx:
                R_hat[user_idx[drop_edge_idx], item_idx[drop_edge_idx]] = 0

        # "A" matrix in the paper
        adj_matrix[:self.n_user, self.n_user:] = R_hat
        adj_matrix[self.n_user:, :self.n_user] = R_hat.T
        adj_matrix = adj_matrix.todok()  # lil_matrix -> dok_matrix


        def norm_adj_single(adj):
            # D^-(1/2) * A * D^-(1/2)
            row_sum = np.array(adj.sum(1))  # sum each row
            d_half = np.power(row_sum, -0.5).flatten()
            d_half[np.isinf(d_half)] = 0.
            d_half_matrix = sp.diags(d_half)

            result = d_half_matrix.dot(adj).dot(d_half_matrix)
            return result.tocoo()

        norm_adj_matrix = norm_adj_single(adj_matrix)
        return norm_adj_matrix.tocsr()


    def negative_pool(self):
        for u in self.train_set.keys():
            candidate_ng_sample = list(set(range(self.n_item)) - set(self.train_set[u]))
            ng_pool = random.sample(candidate_ng_sample, 100)
            self.neg_pools[u] = ng_pool

    def sample(self):
        if self.batch_size <= self.n_user:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_item_for_u(u, num):
            pos_item_for_u = list(self.train_set[u])
            positive_samples = random.sample(pos_item_for_u, num)
            return positive_samples  # [num]

        def sample_neg_item_for_u(u, num):
            pos_item_for_u = list(self.train_set[u])
            negative_pool_for_u = list(set(range(self.n_item)) - set(pos_item_for_u))
            negative_samples = random.sample(negative_pool_for_u, num)
            return negative_samples  # [num]

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_item_for_u(u, 1)
            neg_items += sample_neg_item_for_u(u, 1)

        return users, pos_items, neg_items  # [batch_size], [batch_size], [batch_size]





