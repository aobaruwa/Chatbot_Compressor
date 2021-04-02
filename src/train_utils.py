import argparse
import glob
import logging
import numpy as np
import os
import random
import torch
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.local_rank > 0:
        torch.cuda.manual_seed_all(args.seed)


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
        to_delete = sorted(all_files, key=os.path.getctime)[-1]
        logger.info('Deleting the oldest ckpt {}'.format(to_delete))
        # if 'medium' not in to_delete and 'large' not in to_delete:
        os.remove(to_delete)
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

def set_lr(optimizer, step, schedule, lr, n_embd, tot_steps,
           warmup_steps=700, warmup_proportion=0.1):

    if schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

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

def CEloss(logits, labels):
    """ Evaluate batch CEloss and ppl """
    loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
    loss1 = loss_fct1(logits.view(-1, logits.size(-1)),
                      labels.view(-1))
    loss1 = loss1.view(labels.size(0), labels.size(1))
    label_size = torch.sum(labels != -1, dim=1).type(loss1.type())
    loss = torch.sum(loss1)/torch.sum(label_size)
    ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float()
                    / label_size.float()))
    # ppl = torch.mean(torch.exp(torch.sum(loss1, dim=1)/label_size))
    return loss, ppl
