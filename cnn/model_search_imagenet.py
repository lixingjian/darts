import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, residual_wei, shrink_channel):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.reduction_prev = reduction_prev
    self.residual_wei = residual_wei
    self.shrink_channel = shrink_channel
    if reduction:
        self.residual_reduce = FactorizedReduce(C_prev, C * multiplier)
    elif reduction_prev:
        self.residual_reduce = FactorizedReduce(C_prev_prev, C * multiplier)
    self.residual_norm = nn.ReLU()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0p = self.preprocess0(s0)
    s1p = self.preprocess1(s1)
    states = [s0p, s1p]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    if self.shrink_channel:
        out = None
        for s in states[-self._multiplier:]:
            if out is None:
                out = s
            else:
                out = out + s
    else:
        out = torch.cat(states[-self._multiplier:], dim=1)
    
    '''
    print('shape info')
    print('output', out.shape)
    print('origin', s0.shape, s1.shape)
    print('preproc', s0p.shape, s1p.shape)
    print('normal', self.residual_norm(s0).shape, self.residual_norm(s1).shape)
    if self.reduction:
        print('reduce all', self.residual_reduce(s0).shape, self.residual_reduce(s1).shape)
    elif self.reduction_prev:
        print('reduce s0', self.residual_reduce(s0).shape)
    print('end')
    '''

    if self.reduction:
        out = out + self.residual_wei * self.residual_reduce(s0)
        out = out + self.residual_wei * self.residual_reduce(s1)
    elif self.reduction_prev:
        if s1.shape[1] < out.shape[1]:
            s1 = s1.repeat(1, out.shape[1] // s1.shape[1], 1, 1)
        out = out + self.residual_wei * self.residual_reduce(s0)
        out = out + self.residual_wei * self.residual_norm(s1)
    else:
        if s0.shape[1] < out.shape[1]:
            s0 = s0.repeat(1, out.shape[1] // s0.shape[1], 1, 1)
        if s1.shape[1] < out.shape[1]:
            s1 = s1.repeat(1, out.shape[1] // s1.shape[1], 1, 1)
        out = out + self.residual_wei * self.residual_norm(s0)
        out = out + self.residual_wei * self.residual_norm(s1)
    return out

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=4, residual_wei=1, shrink_channel=False):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.residual_wei = residual_wei
    self.shrink_channel = shrink_channel

    '''
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    '''

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2), 
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )   

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C),
    ) 
    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, residual_wei, shrink_channel)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev = C_prev
      C_prev = C_curr if shrink_channel else multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion,4, 4, 4, self.residual_wei, self.shrink_channel).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    #s0 = s1 = self.stem(input)
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

