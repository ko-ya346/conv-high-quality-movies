import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append('./')
from src.utils import get_line_token


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)

    parser.add_argument('--debug', action='store_true',
                        help='If add it, run with debugging mode (no record and stop one batch per epoch')
    # model setting
    parser.add_argument('--model', type=str, default='srcnn', help='model architecture name')
    parser.add_argument('--upscale', type=int, default=2, help='upscale factor')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function', choices=['mse', 'mse-clop'])

    # dataset setting
    parser.add_argument('--aspect_ratio', type=str, default='4-3', help='aspect ratio')
    parser.add_argument('--dataset', type=str, default='high_2', help='dataset name')
    parser.add_argument('--valid', type=str, default='1-2', help='validation dataset name')

    # optimizer settings
#    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Name')
#    parser.add_argument('--lr', type=float, default=.1, help='learning rate')
#    parser.add_argument('--decay', type=float, default=1e-8, help='weight decay')
#    parser.add_argument('--final_lr', type=float, default=.1,
#                        help='final learning rate (only activa on `optimizer="adabound"`')

    # run settings
#    parser.add_argument('--workers', type=int, default=4, help='workers for dataset parallel')
    parser.add_argument('--batch', type=int, default=128, help='training batch size')
    return vars(parser.parse_args())

class CFG:
    args = get_arguments()
    aspect_ratio = args.get('aspect_ratio')
    dataset = args.get('dataset', None)
    upscale = args.get('upscale', None)

    valid = args.get('valid', None)
    model = args.get('model', None)

    xsize = 720 
    ysize = 960
    low_xsize = xsize // upscale
    low_ysize = ysize // upscale
    batch_size = args.get('batch', None)
    epoch = 10
