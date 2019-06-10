import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
    
def Binarize_pos(tensor, tau=0.):
    with torch.no_grad():
        temp = torch.gt(tensor - tau, 0).float() - tensor
    return tensor + temp

def Binarize_neg(tensor, tau=0.):
    with torch.no_grad():
        temp = torch.lt(tensor + tau, 0).float() + tensor
    return -tensor + temp


class TernaryLinear(nn.Linear):
    
    def __init__(self, *kargs, **kwargs):
        super(TernaryLinear, self).__init__(*kargs, **kwargs)
        self.tau = 0.5
        self.weight.data = self.weight_init()
        
    def weight_init(self):
        self.weight.data.uniform_(-1.,1.)
        return self.weight.data
        
    def forward(self, input):
        if input.size(1) != 784:
#            input.data = Binarize_pos(input.data)
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'ternary'):
            self.weight.ternary = True
        weight_pos = Binarize_pos(self.weight, tau=self.tau)
        weight_neg = Binarize_neg(self.weight, tau=self.tau)
        out_pos = nn.functional.linear(input, weight_pos)
        out_neg = nn.functional.linear(input, weight_neg)
        out = out_pos - out_neg
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out
    
    
class TernaryConv2d(nn.Conv2d):
    
    def __init__(self, *kargs, **kwargs):
        super(TernaryConv2d, self).__init__(*kargs, **kwargs)
        self.tau = 0.5
        self.weight.data = self.weight_init()
        
    def weight_init(self):
        self.weight.data.uniform_(-1.,1.)
        return self.weight.data
        
    def forward(self, input):
        if input.size(1) != 3:
#            input.data = Binarize_pos(input.data)
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'ternary'):
            self.weight.ternary = True
        weight_pos = Binarize_pos(self.weight, tau=self.tau)
        weight_neg = Binarize_neg(self.weight, tau=self.tau)
        out_pos = nn.functional.conv2d(input, weight_pos, bias=None, 
                                       stride=self.stride, padding=self.padding, 
                                       dilation=self.dilation, groups=self.groups)
        out_neg = nn.functional.conv2d(input, weight_neg, bias=None, 
                                       stride=self.stride, padding=self.padding, 
                                       dilation=self.dilation, groups=self.groups)
        out = out_pos - out_neg
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out



class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf

