'''
started on 2022/06/06
end on 2022/xx/xx
@author zelo2
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from snack import parameter_setting


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj_plus_I, norm_adj, device, args):
        super(NGCF, self).__init__()
        self.device = device
        self.n_user = n_user
        self.n_item = n_item
        self.norm_adj = norm_adj
        self.norm_adj_plus_I = norm_adj_plus_I

        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.reg_value = eval(args.reg)[0]

        self.embeding_dict, self.weight_dict = self.init_weight()
        self.sp_norm_adj_plus_I = self.convert_coo_matirix_2_sp_tensor(self.norm_adj_plus_I).to(self.device)
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
        # self.user_embedding = nn.Embedding(self.n_user, self.embed_size)
        # self.item_embedding = nn.Embedding(self.n_user, self.embed_size)
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        # nn.init.xavier_uniform_(self.item_embedding.weight)

        '''Transformation Matrix'''
        weight_dict = nn.ParameterDict()
        for k in range(self.layer_num):
            weight_dict.update({'W1_layer%d' % k: nn.Parameter(initializer(torch.empty(self.embed_size,
                                                                                       self.embed_size)))})
            weight_dict.update({'b1_layer%d' % k: nn.Parameter(initializer(torch.empty(1,
                                                                                       self.embed_size)))})
            weight_dict.update({'W2_layer%d' % k: nn.Parameter(initializer(torch.empty(self.embed_size,
                                                                                       self.embed_size)))})
            weight_dict.update({'b2_layer%d' % k: nn.Parameter(initializer(torch.empty(1,
                                                                                       self.embed_size)))})
        return embedding_dict, weight_dict

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

    def forward(self, user, pos_item, neg_item, drop_flag=False):
        A_hat = self.sp_norm_adj_plus_I
        A = self.sp_norm_adj
        embedding_matrix = torch.cat([self.embeding_dict['user_embed'], self.embeding_dict['item_embed']]
                                     , 0)  # [M+N, embedding_size]
        A_hat = A_hat.to(self.device)
        A = A.to(self.device)
        embedding_matrix = embedding_matrix.to(self.device)

        all_embeddings = [embedding_matrix]

        for k in range(self.layer_num):
            # Graph Convolution operation including self connection
            # [M+N, M+N] * [M+N, embed_size] = [M+N, embed_size]

            neighbor_information_with_self_connection = torch.sparse.mm(A_hat, embedding_matrix)

            # MLP transformation
            neighbor_information_with_self_connection = torch.matmul(neighbor_information_with_self_connection,
                                                                     self.weight_dict['W1_layer%d' % k]) + \
                                                        self.weight_dict['b1_layer%d' % k]  # [M+N, embed_size]

            # Graph Convolution operation without self connection
            neighbor_information = torch.sparse.mm(A, embedding_matrix)
            # element-wise product
            neighbor_information = torch.mul(neighbor_information, embedding_matrix)  # [M+N, M+N]

            # MLP transformation
            neighbor_information = torch.matmul(neighbor_information, self.weight_dict['W2_layer%d' % k]) + \
                                   self.weight_dict['b2_layer%d' % k]  # [M+N, embed_size]

            # activation
            embedding_matrix = nn.LeakyReLU(negative_slope=0.2)(neighbor_information_with_self_connection
                                                                + neighbor_information)
            # Message dropout
            if drop_flag:
                embedding_matrix = nn.Dropout(0.1)(embedding_matrix)

            # Normalization
            norm_embeddings = F.normalize(embedding_matrix, p=2, dim=1)  # normalize each row

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        user_embeddings = all_embeddings[:self.n_user, :]
        item_embeddings = all_embeddings[self.n_user:, :]

        user_embeddings = user_embeddings[user, :]
        pos_item_embeddings = item_embeddings[pos_item, :]
        neg_item_embeddings = item_embeddings[neg_item, :]

        return user_embeddings, pos_item_embeddings, neg_item_embeddings  # [batch_size, embed_size * layer_num] * 3

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

 
