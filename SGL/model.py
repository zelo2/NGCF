'''
started on 2022/06/13
end on 2022/xx/xx
@author zelo2
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from snack import parameter_setting


class LightGCN_SSL(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, device, args):
        super(LightGCN_SSL, self).__init__()
        self.device = device
        self.n_user = n_user
        self.n_item = n_item
        self.norm_adj = norm_adj


        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.reg_value = eval(args.reg)[1]
        self.ssl_reg = eval(args.ssl_reg)[1]
        self.ssl_temp = eval(args.ssl_temp)[1]

        self.embeding_dict = self.init_weight()
        self.sp_norm_adj = self.convert_coo_matirix_2_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        '''Embedding with xavier initialization'''
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_embed': nn.Parameter(initializer(torch.empty(self.n_user,
                                                               self.embed_size))),
            'item_embed': nn.Parameter(initializer(torch.empty(self.n_item,
                                                               self.embed_size)))
        })


        return embedding_dict

    def convert_coo_matirix_2_sp_tensor(self, X):
        coo = X.tocoo()
        # coo matrix--((data, (row, column)), shape)
        # data:矩阵中的数据， row, column表示这个数据在哪一行哪一列
        i = torch.LongTensor([coo.row, coo.col])  # [row, column]
        v = torch.from_numpy(coo.data).float()  # data
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        save_probability = 1 - rate
        # torch.rand: 均匀分布采样[0,1]
        # 因此加上它之后，大于1的概率即为 1 - node_dropout_rate
        save_probability += torch.rand(noise_shape)
        dropout_mask = torch.float(save_probability).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape)

        return out * (1. / (1 - rate))  # dropout部分节点，重新正则化。
        # return out

    def forward(self, user, pos_item, neg_item, sub_graph1, sub_graph2, drop_flag=False):

        A = self.sp_norm_adj
        embedding_matrix = torch.cat([self.embeding_dict['user_embed'], self.embeding_dict['item_embed']]
                                     , 0)  # [M+N, embedding_size]


        all_embeddings = embedding_matrix
        all_embeddings_1 = embedding_matrix
        all_embeddings_2 = embedding_matrix

        for k in range(self.layer_num):

            # Graph Convolution operation without self connection
            embedding_matrix = torch.sparse.mm(A, embedding_matrix)

            embedding_matrix_1 = torch.sparse.mm(sub_graph1, embedding_matrix)  # sub_1
            embedding_matrix_2 = torch.sparse.mm(sub_graph2, embedding_matrix)


            # Message dropout
            if drop_flag:
                embedding_matrix = nn.Dropout(0.1)(embedding_matrix)
                embedding_matrix_1 = nn.Dropout(0.1)(embedding_matrix_1)
                embedding_matrix_2 = nn.Dropout(0.1)(embedding_matrix_2)

            # Normalization
            norm_embeddings = F.normalize(embedding_matrix, p=2, dim=1)  # normalize each row
            norm_embeddings_1 = F.normalize(embedding_matrix_1, p=2, dim=1)  # normalize each row
            norm_embeddings_2 = F.normalize(embedding_matrix_2, p=2, dim=1)  # normalize each row

            all_embeddings += embedding_matrix
            all_embeddings_1 += embedding_matrix_1
            all_embeddings_2 += embedding_matrix_2

        all_embeddings /= (self.layer_num + 1)
        user_embeddings = all_embeddings[:self.n_user, :]
        item_embeddings = all_embeddings[self.n_user:, :]

        all_embeddings_1 /= (self.layer_num + 1)
        user_embeddings_1 = all_embeddings_1[:self.n_user, :]
        item_embeddings_1 = all_embeddings_1[self.n_user:, :]

        all_embeddings_2 /= (self.layer_num + 1)
        user_embeddings_2 = all_embeddings_2[:self.n_user, :]
        item_embeddings_2 = all_embeddings_2[self.n_user:, :]

        user_embeddings = user_embeddings[user, :]
        pos_item_embeddings = item_embeddings[pos_item, :]
        neg_item_embeddings = item_embeddings[neg_item, :]

        user_embeddings_1 = user_embeddings_1[user, :]
        pos_item_embeddings_1 = item_embeddings_1[pos_item, :]
        neg_item_embeddings_1 = item_embeddings_1[neg_item, :]
        final_item_embeddings_1 = torch.cat([pos_item_embeddings_1, neg_item_embeddings_1], 0)

        user_embeddings_2 = user_embeddings_2[user, :]
        pos_item_embeddings_2 = item_embeddings_2[pos_item, :]
        neg_item_embeddings_2 = item_embeddings_2[neg_item, :]
        final_item_embeddings_2 = torch.cat([pos_item_embeddings_2, neg_item_embeddings_2], 0)


        og_results = [user_embeddings, pos_item_embeddings, neg_item_embeddings]
        sub1_results = [user_embeddings_1, final_item_embeddings_1]
        sub2_results = [user_embeddings_2, final_item_embeddings_2]

        return og_results, sub1_results, sub2_results

    def bpr_loss(self, users, pos_items, neg_items):
        '''
        :param users: user embeddings [batch_size, embed_size * layer_num]
        :param pos_items: positive item embeddings
        :param neg_items: negative item embeddings
        :return: Bayesian Personalized Ranking loss (BPR loss)
        '''
        pos_inner_product = torch.mul(users, pos_items)
        neg_inner_product = torch.mul(users, neg_items)

        pos_inner_product = torch.sum(pos_inner_product, axis=1)  # sum each row [batch_size]
        neg_inner_product = torch.sum(neg_inner_product, axis=1)

        loss_value = nn.LogSigmoid()(pos_inner_product - neg_inner_product)
        loss_value = -1 * torch.mean(loss_value)

        # L2范式：所有元素的平方和 开根号
        l2_value = torch.norm(users, p=2) ** 2 + torch.norm(pos_items, p=2) ** 2 + torch.norm(neg_items, p=2) ** 2
        l2_value /= 2

        # for k in range(self.layer_num):
        #     l2_value += torch.norm(self.weight_dict['W1_layer%d' % k], p=2) ** 2
        #     l2_value += torch.norm(self.weight_dict['b1_layer%d' % k], p=2) ** 2

        # l2_value /= (2 + self.layer_num * 2)

        l2_value = self.reg_value * l2_value / self.batch_size

        return loss_value + l2_value

    def contrastive_loss(self, sub1_results, sub2_results):
        user_1 = sub1_results[0]  # [user_num, embed_size]
        user_2 = sub2_results[0]

        item_1 = sub1_results[1]
        item_2 = sub2_results[1]

        user_cl_loss, item_cl_loss = 0., 0.
        all_users = torch.cat([user_1, user_2], 0)
        all_items = torch.cat([item_1, item_2], 0)

        for i in range(sub1_results.shape[0]):
            one_user_cl_loss = F.cosine_similarity(all_users[i], all_users, dim=1)  # cos similarity for each row
            one_user_cl_loss[i] = 0
            one_user_cl_loss = torch.exp(one_user_cl_loss / self.ssl_temp)
            one_user_cl_loss = one_user_cl_loss[0] / torch.sum(one_user_cl_loss)
            one_user_cl_loss = torch.log2(one_user_cl_loss) * (-1)
            user_cl_loss += one_user_cl_loss

        for i in range(sub2_results.shape[0]):
            one_item_cl_loss = torch.exp(F.cosine_similarity(all_items[i], all_items, dim=1) / self.ssl_temp)
            one_user_cl_loss[i] = 0
            one_item_cl_loss = one_item_cl_loss[0] / torch.sum(one_item_cl_loss)
            one_item_cl_loss = torch.log2(one_item_cl_loss) * (-1)
            item_cl_loss += one_item_cl_loss

        cl_loss = (user_cl_loss + item_cl_loss) / self.batch_size

        return self.ssl_reg * cl_loss





