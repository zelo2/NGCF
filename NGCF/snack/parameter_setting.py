'''
started on 2022/06/07
end on 2022/xx/xx
@author zelo2
'''

import argparse

def NGCF_parse():
    parser = argparse.ArgumentParser(description='NGCF')

    parser.add_argument('--device', )

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of iteration')
    parser.add_argument('--reg', default='[1e-5]', nargs='?',
                        help='Regualarization paramater lamda')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding size')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--layer_num', type=int, default=2,
                        help='number of GCN layer')
    parser.add_argument('--dropout_node', type=float, default=0.1,
                        help='probability of node dropout')


    return parser.parse_args()



