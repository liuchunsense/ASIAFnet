import os
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data

from progress.bar import Bar
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from args import arg_parser

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)

# 学习率的设置
def adjust_learning_rate(args, optimizer, i_iter, Max_step):
    """Sets the learning rate to the initial LR divided"""

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    lr = lr_poly(args.learning_rate, i_iter, Max_step, 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # optimizer.param_groups[0]['lr'] = lr
    return lr

# 参数类
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    """Create the model and start the training."""

    cudnn.enabled = True
    cudnn.benchmark = True
    
    if not args.object:
        from Data.gazedata import Gaze_not_object as GazePredictData
    else:
        from Data.gazedata import Gaze_object as GazePredictData
    if args.fcn:
        from Network.FCN import GAZE as Model
    else:
        from Network.ASTAFNet import ASTAFNet as Model

    model = Model(importance_is=args.object, rf=args.rf).cuda()

    TrainDataLoader = data.DataLoader(
        GazePredictData(data_path=args.train_path,datatype='train', txt_dir = args.txt_dir,data_dir=args.data_dir,im_or_video=args.image),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if (not os.path.exists(args.save)):
        os.makedirs(args.save)

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()

    scaler = GradScaler()

    for epoch in range(args.epochs):
        train_loss = train(model, TrainDataLoader, optimizer, epoch, scaler)
        torch.save(model.state_dict(), os.path.join(args.save, 'GazePrediction_best.pth'))

def train(model, TrainDataLoader, optimizer, epoch, scaler):
    ''' Train the model. '''
    model.train()

    bar = Bar('{}'.format('Gaze Prediction'), max=len(TrainDataLoader))
    losses = AverageMeter()

    if args.object:
        for index, (frame, target, fixmap, heatmap, bboxs, labels) in enumerate(TrainDataLoader):
            
            frame = frame.cuda()
            target = target.cuda()
            fixmap = fixmap.cuda()
            bboxs = bboxs.cuda()
            heatmap = heatmap.cuda()

            labels = labels.cuda()
            batch = frame.size(0)
            
            # 因为pytorch的roipool的需要，这里需要将目标的位置信息进行转换，变成batch*n*5大小
            bboxs = bboxs.view(batch * bboxs.size(1), 4)
            device, dtype = bboxs.device, bboxs.dtype

            ids = torch.cat(
                [
                    torch.full_like(torch.rand(30, 1), i, dtype=dtype, layout=torch.strided, device=device)
                    for i in range(batch)
                ],
                dim=0,
            )

            rois = torch.cat([ids, bboxs], dim=1).half()

            optimizer.zero_grad()

            lr = adjust_learning_rate(args, optimizer, epoch * len(TrainDataLoader) + index,
                                      args.epochs * len(TrainDataLoader))

            loss, kld_loss, cc_loss, sim_loss, nss_loss, bce_loss, acc = model(frame, target, fixmap, heatmap, rois,
                                                                               labels)

            loss = loss.mean()
            loss.backward()

            optimizer.step()
            if loss < 1000:
                losses.update(float(loss))

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, index, len(TrainDataLoader), phase='train',
                total=bar.elapsed_td, eta=bar.eta_td)

            Bar.suffix = Bar.suffix + \
                         '|{} {:.4f}|{} {:.4f}|{} {:.4f}|{} {:.4f} |{} {:.4f} |{} {:.4f} |{} {:.4f} |{} {:.4f}  \n'. \
                             format('lr', lr, 'loss', loss, 'kld', kld_loss.mean(), 'cc', cc_loss.mean(), 'sim',
                                    sim_loss.mean(),
                                    'nss', nss_loss.mean(), 'bce', bce_loss.mean(), 'acc', acc)

            if (index + 1) % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(args.save, 'GazePrediction_{}_{}.pth'.format(epoch, index)))

            print(Bar.suffix)
            bar.next()

    else:
        for index, (frame, target, fixmap) in enumerate(TrainDataLoader):
            if index < 11999: continue
            if index > 14000: break
            frame = frame.cuda().squeeze(dim=1)
            target = target.cuda()
            fixmap = fixmap.cuda()

            optimizer.zero_grad()

            lr = adjust_learning_rate(args, optimizer, epoch * len(TrainDataLoader) + index,
                                      args.epochs * len(TrainDataLoader))
            with autocast():
                loss, kld_loss, cc_loss, sim_loss, nss_loss = model(frame, target, fixmap)

            loss = loss.mean()
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            if loss < 1000:
                losses.update(float(loss))

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, index, len(TrainDataLoader), phase='train',
                total=bar.elapsed_td, eta=bar.eta_td)

            Bar.suffix = Bar.suffix + \
                         '|{} {:.7f}|{} {:.4f}|{} {:.4f}|{} {:.4f}|{} {:.4f}|{} {:.4f} \n'. \
                             format('lr', lr, 'loss', loss, 'kld', kld_loss, 'cc', cc_loss, 'sim',
                                    sim_loss, 'nss', nss_loss)

            if (index + 1) % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(args.save, 'GazePrediction_{}_{}.pth'.format(epoch, index)))

            print(Bar.suffix)
            bar.next()

    print("{} | epoch :{} | loss :{}".format('train', epoch, losses.avg))
    bar.finish()
    return losses.avg


if __name__ == '__main__':
    main()
