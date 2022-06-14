'''
started on 2022/06/13
end on 2022/xx/xx
@author zelo2
'''

import argparse

# Follow this parameter setting:     https://github.com/wujcan/SGL-Torch
def SGL_parse():
    parser = argparse.ArgumentParser(description='LightGCN')

    parser.add_argument('--device', )

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of iteration')
    parser.add_argument('--reg', default='[1e-4]', nargs='?',
                        help='Regularization parameter lamda_2')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding size')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='number of GCN layer')

    '''Parameters of graph contrastive learning'''
    parser.add_argument('--ssl_rate', nargs='?', default='[0.1, 0.1, 0.4]',
                        help='probability of dropout (yelp2018, amazon-book, amazon-ifashion)')
    parser.add_argument('--ssl_temp', nargs='?', default='[0.2, 0.2, 0.5]',
                        help='temperature parameter')
    parser.add_argument('--ssl_reg', nargs='?', default='[0.1, 0.5, 0.02]',
                        help='regularization parameter for contrastive learning loss')
    parser.add_argument('--aug_type', nargs='?', default='[ED, ND]',
                        help='type of data augmentation (ED-Edge Drop, ND-Node Drop)')


    return parser.parse_args()



