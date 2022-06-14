'''
started on 2022/06/13
end on 2022/xx/xx
@author zelo2
'''

import argparse

def LightGCN_parse():
    parser = argparse.ArgumentParser(description='LightGCN')

    parser.add_argument('--device', )

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of iteration')
    parser.add_argument('--reg', default='[1e-4]', nargs='?',
                        help='Regualarization paramater lamda')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding size')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='number of GCN layer')
    parser.add_argument('--dropout_node', type=float, default=0.1,
                        help='probability of node dropout')


    return parser.parse_args()


