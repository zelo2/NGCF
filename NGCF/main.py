'''
started on 2022/06/08
end on 2022/xx/xx
@author zelo2
'''

import model
from snack import data_load
from snack import parameter_setting
import torch
import numpy as np
import sklearn.metrics as metrics

if __name__ == '__main__':
    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

    parser_ngcf = parameter_setting.NGCF_parse()

    path = ['../Data/amazon-book', '../Data/gowalla']
    path = path[1]
    batch_size = parser_ngcf.batch_size
    data = data_load.Data(path, batch_size)
    norm_adj, norm_adj_plus_I = data.creat_adj_mat()
    print("User Number:", data.n_user)
    print("Item Number:", data.n_item)
    print("Interactions:", data.n_train + data.n_test)
    print("Density:", (data.n_train + data.n_test) / (data.n_user * data.n_item))

    net = model.NGCF(data.n_user, data.n_item, norm_adj_plus_I, norm_adj, device, parser_ngcf)
    net = net.to(device)

    '''Train'''
    optimizer = torch.optim.Adam(net.parameters(), lr=parser_ngcf.lr)
    loss_loger, recall_loger, ndcg_loger = [], [], []

    for epoch in range(parser_ngcf.epoch):  # parser_ngcf.epoch
        print("Train")
        loss = 0
        n_batch = data.n_user // batch_size + 1
        for batch_iteration in range(n_batch):
            users, pos_items, neg_items = data.sample()  # [batch_size] * 3
            users = torch.LongTensor(users).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)

            user_embeddings, pos_item_embeddings, neg_item_embeddings = net(users, pos_items, neg_items, drop_flag=True)
            batch_loss = net.bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        loss_loger.append(loss)
        print("epoch:%d BPR loss:%d" % (epoch, loss))

        '''Test/Validation'''
        if epoch > 19 and (epoch + 1) % 20 == 0:
            with torch.no_grad():
                print("Test")
                test_recall = 0
                k = 20  # Recall@20, NDCG@20
                ndcg_k_collection = []
                test_batch_size = batch_size * 2
                num_batch = data.n_test // batch_size + 1

                for test_user in data.test_set.keys():
                    test_item_sequence = data.test_set[test_user]
                    train_item_sequence = data.train_set[test_user]
                    item_set = range(data.n_item)

                    test_user_embeddings, all_item_embeddings, _ = net(test_user, item_set, [], drop_flag=False)

                    # all_item_embeddings = net.embeding_dict['item_embed']
                    all_item_embeddings[train_item_sequence, :] = 0  # delete training data
                    ratings = torch.matmul(test_user_embeddings, all_item_embeddings.T).cpu()  # [item_num]
                    ratings = np.array(ratings)
                    rating_index = np.argsort(ratings)
                    rating_index_max20 = rating_index[-20:]
                    ratings_max20 = ratings[rating_index_max20]  # top-k ratings


                    '''NDCG@k'''
                    ndcg_k_rating = torch.matmul(test_user_embeddings, all_item_embeddings[test_item_sequence].T).cpu()
                    ndcg_k_rating = np.array(ndcg_k_rating)
                    dcg_k = np.sum(ndcg_k_rating / np.log2(np.arange(2, len(ndcg_k_rating) + 2)))
                    if len(test_item_sequence) < k:
                        ideal_rating = [1.] * len(test_item_sequence) + [0.] * (k - len(test_item_sequence))
                    else:
                        ideal_rating = [1.] * k
                    dcg_k_max = np.sum(ideal_rating / np.log2(np.arange(2, k + 2)))
                    ndcg_k = dcg_k / dcg_k_max
                    ndcg_k_collection.append(ndcg_k)


                    for rec_item_id in rating_index_max20:
                        if rec_item_id in test_item_sequence:
                            test_recall += 1

                '''
                How to compute these evalution metrics?
                http://www.javashuo.com/article/p-npsntsvi-mw.html
                https://zhuanlan.zhihu.com/p/136199536
                '''

                '''Recall@20'''
                recall_20 = test_recall / data.n_test
                recall_loger.append(recall_20)

                '''NDCG@20'''
                ndcg_20_final = np.mean(np.array(ndcg_k_collection))

                print("Recall@20:", recall_20)
                print("NDCG@20:", ndcg_20_final)


