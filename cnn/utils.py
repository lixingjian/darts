import os
import math
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

class AverageMeterTime(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def set_lr(optimizer, T_cur, T_total, lr_init):
  lr_adj = 0.5 * lr_init * (1 + math.cos(math.pi * T_cur / T_total)) 
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr_adj
  return lr_adj

def set_group_weight(module, bn_no_wd=True, bias_no_wd=True):
  if not (bn_no_wd or bias_no_wd):
    return module.parameters()

  group_decay = []
  group_no_decay = []
  
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if bias_no_wd and (m.bias is not None):
        group_no_decay.append(m.bias)
      else:
        group_decay.append(m.bias)
    elif isinstance(m, nn.Conv2d):
      group_decay.append(m.weight)
      if bias_no_wd and (m.bias is not None):
        group_no_decay.append(m.bias)
      else:
        group_decay.append(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
      if m.bias is not None:
        if bn_no_wd:
          group_no_decay.append(m.weight)
        else:
          group_decay.append(m.weight)
      if m.bias is not None:
        if bn_no_wd:
          group_no_decay.append(m.bias)
        else:
          group_decay.append(m.bias)
    else:
      for w in m.parameters():
        group_decay.append(w)
  group_decay = list(set(module.parameters()) - set(group_no_decay))
  print('decay params: %d' % len(group_decay))
  print('no decay params: %d' % len(group_no_decay))
  print('all params: %d' % len(list(module.parameters())))
  assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
  groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
  return groups

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  d = {}
  for k, v in torch.load(model_path).items():
    if k.startswith('module.'):
      d[k[len('module.'):]] = v
    else:
      d[k] = v
  model.load_state_dict(d)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
      os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

