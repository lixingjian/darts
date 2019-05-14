import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network
from utils import ProgressMeter, AverageMeterTime

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='./data/image-net', help='location of the data corpus')
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='num of training epochs')
parser.add_argument('--ngpu', type=int, default=8, help='num of gpus')
#parser.add_argument('--early_stop', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')

parser.add_argument('--lr_strategy', type=str, default='cos', choices=['cos'])
parser.add_argument('--bn_no_wd', action='store_true', default=False)
parser.add_argument('--bias_no_wd', action='store_true', default=False)
parser.add_argument('--residual_wei', type=float, default=2, help='weight decay for arch encoding')
parser.add_argument('--shrink_channel', action='store_true', default=False, help='weight decay for arch encoding')
args = parser.parse_args()
print(torch.__version__)

if args.save == '':
    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000
#args.ngpu = torch.cuda.device_count()
print('args.ngpu = %d' % args.ngpu)

args.batch_size *= args.ngpu
nworker = 4 * args.ngpu
args.learning_rate *= (args.batch_size / 128.0)

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  #torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device num = %d' % args.ngpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, args.residual_wei, args.shrink_channel)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
    #model = nn.parallel.DistributedDataParallel(model).cuda()
  else:
    model = model.cuda()
  
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    #model.parameters(),
    utils.set_group_weight(model, args.bn_no_wd, args.bias_no_wd),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  resume = os.path.join(args.save, 'checkpoint.pth.tar')
  if os.path.exists(resume):
    print("=> loading checkpoint %s" % resume)
    #checkpoint = torch.load(resume)
    checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda(0))
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.state_dict()['state'] = checkpoint['optimizer']['state']
    print('=> loaded checkpoint epoch %d' % args.start_epoch)
    if args.start_epoch >= args.epochs:
        print('training finished')
        sys.exit(0)

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(args.image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(int((256.0 / 224) * args.image_size)),
      transforms.CenterCrop(args.image_size),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nworker)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nworker)

  best_acc_top1 = 0
  for epoch in range(args.start_epoch, args.epochs):
    if args.lr_strategy == 'cos':
      lr = utils.set_lr(optimizer, epoch, args.epochs, args.learning_rate)
    #elif args.lr_strategy == 'step':
    #  scheduler.step()
    #  lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    if args.parallel:
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    else:
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, epoch)
    logging.info('train_acc %f', train_acc)

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': train_acc,
      'optimizer' : optimizer.state_dict(),
      }, False, args.save)

    #if epoch >= args.early_stop:
    #  break

  valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
  logging.info('valid_acc_top1 %f', valid_acc_top1)
  logging.info('valid_acc_top5 %f', valid_acc_top5)

def train(train_queue, model, criterion, optimizer, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  batch_time = AverageMeterTime('Time', ':6.3f')
  data_time = AverageMeterTime('Data', ':6.3f')
  progress = ProgressMeter(len(train_queue), batch_time, data_time, prefix="Epoch: [{}]".format(epoch))
  model.train()

  torch.cuda.synchronize()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    torch.cuda.synchronize()
    data_time.update(time.time() - end)
    end = time.time()
    target = target.cuda()
    #target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)

    torch.cuda.synchronize()
    batch_time.update(time.time() - end)
    end = time.time()
    
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f data_time:%.3f/%.3f batch_time:%.3f/%.3f speed:%.3f/%.3f', step, objs.avg, top1.avg, top5.avg, data_time.val, data_time.avg, batch_time.val, batch_time.avg, args.batch_size/(data_time.val+batch_time.val), args.batch_size/(data_time.avg+batch_time.avg))
    
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
