import pickle
import os
import argparse
from datetime import datetime


def argparser():
    parser = argparse.ArgumentParser()
    # main parts
    parser.add_argument('-m', '--mode', metavar='M', type=str, default='train', choices=['train', 'train_emv', 'test'],
                        help='train or train_emv or test')
    parser.add_argument('-b', '--batch', metavar='B', type=int, default=16, help='batch size')
    parser.add_argument('-t', '--task_n', metavar='T', type=int, default=100,
                        help='number of cities(nodes), time sequence, default: 20')
    parser.add_argument('--server_load', type=int, default=5, help='server load')
    parser.add_argument('-s', '--steps', metavar='S', type=int, default=2000,
                        help='training steps(epochs), default: 15000')

    # details
    parser.add_argument('-e', '--embed', metavar='EM', type=int, default=128, help='embedding size')
    parser.add_argument('-hi', '--hidden', metavar='HI', type=int, default=128, help='hidden size')
    parser.add_argument('-c', '--clip_logits', metavar='C', type=int, default=10,
                        help='improve exploration; clipping logits')
    parser.add_argument('-st', '--softmax_T', metavar='ST', type=float, default=1.0,
                        help='might improve exploration; softmax temperature default 1.0 but 2.0, 2.2 and 1.5 might yield better results')
    parser.add_argument('-o', '--optim', metavar='O', type=str, default='Adam', help='torch optimizer')
    parser.add_argument('-minv', '--init_min', metavar='MINV', type=float, default=-0.08,
                        help='initialize weight minimun value -0.08~')
    parser.add_argument('-maxv', '--init_max', metavar='MAXV', type=float, default=0.08,
                        help='initialize weight ~0.08 maximum value')
    parser.add_argument('-ng', '--n_glimpse', metavar='NG', type=int, default=1, help='how many glimpse function')
    parser.add_argument('-np', '--n_process', metavar='NP', type=int, default=3,
                        help='how many process step in critic; at each process step, use glimpse')
    parser.add_argument('-dt', '--decode_type', metavar='DT', type=str, default='sampling',
                        choices=['greedy', 'sampling'], help='how to choose next task in actor model')

    parser.add_argument('--alphaa', type=float, default=1, help='weight for load impact')
    parser.add_argument('--beta', type=float, default=1, help='weight for priority impact')
    parser.add_argument('--gama', type=float, default=1, help='weight for timeout impact')

    # train, learning rate
    parser.add_argument('--lr', metavar='LR', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--is_lr_decay', action='store_false', help='flag learning rate scheduler default true')
    parser.add_argument('--lr_decay', metavar='LRD', type=float, default=0.96,
                        help='learning rate scheduler, decay by a factor of 0.96 ')
    parser.add_argument('--lr_decay_step', metavar='LRDS', type=int, default=500,
                        help='learning rate scheduler, decay every 5000 steps')

    # inference
    parser.add_argument('-ap', '--act_model_path', metavar='AMP', default='./Pt/train20_0717_13_20_step14999_act.pt',
                        type=str, help='load actor model path')
    parser.add_argument('--seed', metavar='SEED', type=int, default=123,
                        help='random seed number for inference, reproducibility')
    parser.add_argument('-al', '--alpha', metavar='ALP', type=float, default=0.99, help='alpha decay in active search')

    # path
    parser.add_argument('--islogger', action='store_false', help='flag csv logger default true')
    parser.add_argument('--issaver', action='store_false', help='flag model saver default true')
    parser.add_argument('-md', '--model_dir', metavar='MD', type=str, default='./Pt/', help='model save dir')
    parser.add_argument('-pd', '--pkl_dir', metavar='PD', type=str, default='./Pkl/', help='pkl save dir')

    # GPU
    parser.add_argument('-cd', '--cuda_dv', metavar='CD', type=str, default='0',
                        help='os CUDA_VISIBLE_DEVICE, default single GPU')
    args = parser.parse_args()
    return args


class Config():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.dump_date = datetime.now().strftime('%m%d_%H_%M')
        self.task = '%s%d' % (self.mode, self.task_n)
        self.pkl_path = self.pkl_dir + '%s.pkl' % (self.task)
        self.n_samples = self.batch * self.steps
        for x in [self.model_dir, self.pkl_dir]:
            os.makedirs(x, exist_ok=True)


def print_cfg(cfg):
    print(''.join('%s: %s\n' % item for item in vars(cfg).items()))


def dump_pkl(args, verbose=True):
    cfg = Config(**vars(args))
    with open(cfg.pkl_path, 'wb') as f:
        pickle.dump(cfg, f)
        print('--- save pickle file in %s ---\n' % cfg.pkl_path)
        if verbose:
            print_cfg(cfg)


def load_pkl(pkl_path, verbose=True):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError('pkl_path')
    with open(pkl_path, 'rb') as f:
        cfg = pickle.load(f)
        if verbose:
            print_cfg(cfg)
        os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
    return cfg


def pkl_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', metavar='P', type=str,
                        default='Pkl/train100.pkl', help='pkl file name')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = argparser()
    dump_pkl(args)
