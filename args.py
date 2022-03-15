import time
import argparse
arg_parser = argparse.ArgumentParser(description='Gaze Prediction')

# experiment related
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='checkpoint/save_astaf_rfb',
                       type=str, metavar='SAVE',
                       help='directory path to save model')
exp_group.add_argument('--seed', default=42, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default='0', type=str, help='GPU available.')
exp_group.add_argument('--epochs', default=1, type=int, help='the number of epochs')
exp_group.add_argument('--batch_size', default=12, type=int, help='batch size')
exp_group.add_argument('--num_workers', default=16, type=int, help='number of workers')
exp_group.add_argument('--group', default=0, type=int, help='the start epoch')
exp_group.add_argument('--image', default=False,help='input is image or video?')
exp_group.add_argument('--object', default=True,help='estimate the saliency of objects?')
exp_group.add_argument('--fcn', default=False,help='the model is FCN-50')
exp_group.add_argument('--rf', default='rfb',help='[aac, rfb, aspp, ppm]')
exp_group.add_argument('--txt_dir', default='/home/liq/Desktop/DADA/importance/',help='object saliency information')
# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--train_path', metavar='DIR', default='/home/liq/Desktop/DADA/dataset/train_file.json',
                        help='path to dataset (default: data)')
data_group.add_argument('--val_path', metavar='DIR', default='/home/liq/Desktop/DADA/dataset/test_file.json',
                        help='path to dataset (default: data)')
data_group.add_argument('--data_dir', metavar='DIR', default='/home/liq/Desktop/DADA/NEW_DATA/',
                        help='path to dataset (default: data)')


# optimization related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')

optim_group.add_argument('--learning_rate', default=1e-3, type=int, metavar='N',
                         help='learning rate')
optim_group.add_argument('--momentum', default=0.9, type=int, metavar='N',
                         help='learning rate')
optim_group.add_argument('--weight_decay', default=0.0005, type=int, metavar='N',
                         help='learning rate')
optim_group.add_argument('--mixed_precision',dest='mixed_precision',action='store_true', default=True)

