from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_arguments():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="If add it, run with debugging mode (no record and stop one batch per epoch",
    )
    # model setting
    parser.add_argument(
        "--model", type=str, default="srcnn", help="model architecture name"
    )
    parser.add_argument("--upscale", type=int, default=3, help="upscale factor")
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function",
        choices=["mse", "mse-clop"],
    )

    # dataset setting
    parser.add_argument(
        "--dataset", type=str, default="T91", help="dataset name"
    )
    parser.add_argument(
        "--valid", type=str, default="1-2", help="validation dataset name"
    )

    # optimizer settings
    #    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Name')
    #    parser.add_argument('--lr', type=float, default=.1, help='learning rate')
    #    parser.add_argument('--decay', type=float, default=1e-8, help='weight decay')
    #    parser.add_argument('--final_lr', type=float, default=.1,
    #                        help='final learning rate (only activa on `optimizer="adabound"`')

    # run settings
    #    parser.add_argument('--workers', type=int, default=4, help='workers for dataset parallel')
    parser.add_argument(
        "--batch", type=int, default=128, help="training batch size"
    )
    return vars(parser.parse_args())


class CFG:
    args = get_arguments()
    name_experiment = "pix6"
    debug = False

    train_dataset = "T91"
    valid_dataset = "Set5"
    inference_dataset = "rori"
    shrink_scale = 3

    model = "bnsrcnn"
    max_size = 128
    total_samples = 1000
    input_upsample = True

    inference_dataset = "rori"
    xsize = 240
    ysize = 324
    up_scale = 3
    #
    # optimizerとschedulerのパラメータがよく分からん
    #
    lr = 0.01
    weight_decay = 1e-8
    lr_step = 15
    lr_gamma = 0.1

    batch_size = 128
    epoch = 100

    # pix2pixのbceとmaeの係数
    # 大きいほど元画像に近くなる（論文では100）
    LAMBD = 200.0
    gan_mode = "vanilla"
