import argparse
import glob
import logging
import numpy as np
import os
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.local_rank > 0:
        torch.cuda.manual_seed_all(args.seed)


def bool_flag(s: str) ->bool:
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    elif s.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def _save_checkpoint(model, optimizer, global_step, args):
    model_dir = os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'GPT2-pretrain-step-{global_step+1}.pkl')
   
    model_state_dict = model.module.state_dict() \
            if isinstance(model, torch.nn.DataParallel) \
            else model.state_dict()
    state = {
            "steps": global_step,
            "model_state": model_state_dict,
            "optimizer_state": optimizer.state_dict(),
            "amp_state": amp.state_dict() if args.use_fp16 else None
        }
    all_files = glob.glob('{}/*.pkl'.format(os.path.dirname(args.ckpt_file)))
    if not all_files:
        raise FileNotFoundError(
            "No checkpoint found in directory {}.".format(
                os.path.dirname(args.ckpt_file)))
    if all_files:
        srtd_files = sorted(all_files, key=os.path.getctime)
        #logger.info('Deleting the oldest ckpt {}'.format(to_delete))
        # if 'medium' not in to_delete and 'large' not in to_delete:
        oldest_ckpt = srtd_files[-1]
        if 'medium' in oldest_ckpt and len(srtd_files) >= 2:
            os.remove(srtd_files[-2])
        elif 'medium' not in oldest_ckpt:
            os.remove(oldest_ckpt)
    logger.info('saving new model to %s' % model_dir)
    torch.save(state,  model_path)


def _load_checkpoint(ckpt_file, model, optimizer, args):
    logger.info('loading checkpoint from {}'.format(ckpt_file))
    # load state dicts
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])

    if args.use_fp16 and 'amp_state' in ckpt.keys():
        try:
            from apex import amp
            amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Apex is not installed.")


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)))


def noamwd_decay(step, warmup_steps,
                 model_size, rate=0.5, decay_steps=1000, start_step=500):
    """Learning rate schedule optimized for huge batches
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)) *
        rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)
    
SCHEDULES = {
    'warmup_cosine': None,
    'warmup_constant': None,
    'warmup_linear': warmup_linear,
}

class Adam(Optimizer):

    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(Adam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


def set_lr(optimizer, step, schedule, lr, n_embd, tot_steps,
           warmup_steps=300, warmup_proportion=0.1):

    if schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step



def CEloss(logits, labels):
    """ Evaluate batch CEloss and ppl """
    loss_fct1 = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss1 = loss_fct1(logits.view(-1, logits.size(-1)),
                      labels.view(-1))
    loss1 = loss1.view(labels.size(0), labels.size(1))
    label_size = torch.sum(labels != -1, dim=1).type(loss1.type())
    loss = torch.sum(loss1)/torch.sum(label_size)
    if len(loss.size()) > 1: # distributed training
        ppl = torch.exp(torch.mean(torch.sum(loss1.mean().detach().cpu().item(), dim=1).float()
                    / label_size.float()))
    else:
        ppl = torch.exp(torch.mean(torch.sum(loss1.detach(), dim=1).float()
                    / label_size.float()))
    # ppl = torch.mean(torch.exp(torch.sum(loss1, dim=1)/label_size))
    return loss, ppl
