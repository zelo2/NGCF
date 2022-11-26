'''
started on 2022/11/24
end on 2022/xx/xx
@author zelo2
'''
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from snack import parameter_setting, data_NGCF
from dgl.utils import expand_as_pair
import tqdm


class NGCF_layer(nn.Module):
    def __init__(self, args):
        super(NGCF_layer, self).__init__()

        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num

        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2)
        self.linear_layer_4_gcn = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.linear_layer_4_infor_enhance = nn.Linear(self.embed_size, self.embed_size, bias=False)
        nn.init.xavier_uniform_(self.linear_layer_4_gcn.weight)
        nn.init.xavier_uniform_(self.linear_layer_4_infor_enhance.weight)

    def forward(self, g, feature):
        '''
        :param g: dgl graph
        :param feature: (user, item) embedding matrix
        :return: embedding matrix
        '''

        def ngcf_msg_func1(edges):  # gcn-like
            return {'m_1': edges.dst['emd'] / (torch.sqrt(edges.src['in_d']) * torch.sqrt(edges.dst['in_d']))}

        def ngcf_msg_func2(edges):  # point-wise product
            return {'m_2': torch.mul(edges.dst['emd'], edges.src['emd']) / (
                    torch.sqrt(edges.src['in_d']) * torch.sqrt(edges.dst['in_d']))}

        g.ndata['emd'] = feature
        g.update_all(ngcf_msg_func1, dgl.function.sum('m_1', 'result_1'))
        g.update_all(ngcf_msg_func2, dgl.function.sum('m_2', 'result_2'))

        result = self.linear_layer_4_gcn(g.ndata['result_1']) + self.linear_layer_4_infor_enhance(g.ndata['result_2'])
        result = result + self.linear_layer_4_gcn(feature)
        result = self.leakey_relu(result)
        result = F.normalize(result, p=2, dim=1)

        return result


class NGCF_dgl(nn.Module):
    def __init__(self, n_user, n_item, R, device, args):
        super(NGCF_dgl, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.device = device
        self.n_user = n_user
        self.n_item = n_item
        self.R = R.tocoo()

        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.reg_value = eval(args.reg)[0]

        # Initialize the graph
        self.g = dgl.to_bidirected(dgl.graph((self.R.row, self.R.col + self.n_user)))
        self.g.ndata['in_d'] = self.g.out_degrees().reshape(-1, 1)
        self.g = self.g.to(self.device)
        self.feature = nn.Parameter(initializer(torch.empty(self.n_user + self.n_item, self.embed_size)),
                                    requires_grad=True)

        self.ngcf_layer_list = []
        for i in range(self.layer_num):
            self.ngcf_layer_list.append(NGCF_layer(args=args).to(self.device))

    def forward(self, user, pos_item, neg_item):
        '''
        :param user: [batch_size] user id
        :param pos_item: [batch_size] positive item id
        :param neg_item: [batch_size] negative item id
        :return: Learned representations of user and item via NGCF
        '''
        with self.g.local_scope():
            all_embeddings = [self.feature]  # layer 0 embeddings
            input_feature = None
            for current_layer in range(self.layer_num):
                if input_feature is None:
                    input_feature = self.feature
                    output_embeddings = self.ngcf_layer_list[current_layer](self.g, input_feature)
                    input_feature = output_embeddings
                else:
                    output_embeddings = self.ngcf_layer_list[current_layer](self.g, input_feature)
                    input_feature = output_embeddings

                all_embeddings += [output_embeddings]

            all_embeddings = torch.cat(all_embeddings, 1)
            user_embeddings = all_embeddings[:self.n_user, :]
            item_embeddings = all_embeddings[self.n_user:, :]

            user_embeddings = user_embeddings[user, :]
            pos_item_embeddings = item_embeddings[pos_item, :]
            neg_item_embeddings = item_embeddings[neg_item, :]

            return user_embeddings, pos_item_embeddings, neg_item_embeddings

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


if __name__ == '__main__':
    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
    print("Available Device:", device)

    parser_ngcf = parameter_setting.NGCF_parse()

    path = ['../Data/amazon-book', '../Data/gowalla']
    path = path[1]
    batch_size = parser_ngcf.batch_size
    data = data_NGCF.Data(path, batch_size)
    print("User Number:", data.n_user)
    print("Item Number:", data.n_item)
    print("Interactions (Train + Test):", data.n_train + data.n_test)
    print("Density:", (data.n_train + data.n_test) / (data.n_user * data.n_item))

    net = NGCF_dgl(data.n_user, data.n_item, data.R, device, parser_ngcf)
    net = net.to(device)
    print("NGCF Net is on:", net.device)

    '''Train'''
    optimizer = torch.optim.Adam(net.parameters(), lr=parser_ngcf.lr)
    loss_loger, recall_loger, ndcg_loger = [], [], []

    for epoch in range(parser_ngcf.epoch):  # parser_ngcf.epoch
        print("Train")
        loss = 0.
        n_batch = data.n_train // batch_size + 1
        print("n_batch", n_batch)
        for batch_iteration in tqdm.tqdm(range(n_batch)):
            users, pos_items, neg_items = data.sample()  # [batch_size] * 3
            # users = torch.LongTensor(users).to(device)
            # pos_items = torch.LongTensor(pos_items).to(device)
            # neg_items = torch.LongTensor(neg_items).to(device)

            user_embeddings, pos_item_embeddings, neg_item_embeddings = net(users, pos_items, neg_items)
            batch_loss = net.bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        loss_loger.append(loss)
        print("epoch:%d BPR loss:%f" % (epoch, loss))

        '''Test/Validation'''
        if epoch > 19 and (epoch + 1) % 20 == 0:
            # if epoch > 5:
            with torch.no_grad():
                print("Test")

                k = 20  # Recall@20, NDCG@20
                ndcg_k_collection = []
                recall_k_collection = []
                test_batch_size = batch_size * 2
                num_batch = data.n_test // batch_size + 1

                for test_user in data.test_set.keys():
                    test_ratings = []
                    test_item_sequence = data.test_set[test_user]
                    train_item_sequence = data.train_set[test_user]
                    item_set = range(data.n_item)

                    test_user_embeddings, all_item_embeddings, _ = net(test_user, item_set, [])

                    # all_item_embeddings = net.embeding_dict['item_embed']
                    all_item_embeddings[train_item_sequence, :] = 0  # delete training data
                    ratings = torch.matmul(test_user_embeddings, all_item_embeddings.T).cpu()  # [item_num]
                    # print(ratings)
                    ratings = np.array(ratings)
                    rating_index = np.argsort(ratings)
                    rating_index_max20 = rating_index[-20:]
                    ratings_max20 = ratings[rating_index_max20]  # top-k ratings

                    '''Rating computation '''
                    for top_k_item in rating_index_max20:
                        if top_k_item in test_item_sequence:
                            test_ratings.append(1.0)
                        else:
                            test_ratings.append(0)

                    '''NDCG@k'''
                    dcg_k = np.sum(test_ratings / np.log2(np.arange(2, k + 2)))
                    if len(test_item_sequence) < k:
                        ideal_rating = [1.] * len(test_item_sequence) + [0.] * (k - len(test_item_sequence))
                    else:
                        ideal_rating = [1.] * k
                    dcg_k_max = np.sum(ideal_rating / np.log2(np.arange(2, k + 2)))
                    ndcg_k = dcg_k / dcg_k_max
                    ndcg_k_collection.append(ndcg_k)

                    '''Recall@20'''
                    recall_20 = np.sum(np.asfarray(test_ratings)) / len(test_item_sequence)
                    recall_k_collection.append(recall_20)
                '''
                How to compute these evalution metrics?
                http://www.javashuo.com/article/p-npsntsvi-mw.html
                https://zhuanlan.zhihu.com/p/136199536
                '''

                '''Recall@20'''
                recall_20_final = np.mean(np.array(recall_k_collection))
                recall_loger.append(recall_20_final)

                '''NDCG@20'''
                ndcg_20_final = np.mean(np.array(ndcg_k_collection))
                ndcg_loger.append(ndcg_20_final)

                print("Recall@20:", recall_20_final)
                print("NDCG@20:", ndcg_20_final)

    print("Recall@20:", recall_loger)
    print("NDCG@20:", ndcg_loger)
