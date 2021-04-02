# coding: utf-8
"""
Training module
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
from types import SimpleNamespace
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead
)
from eval_utils import eval_model
from loader import Loader, DistLoader
from train_utils import (
    bool_flag,
    set_seed,
    set_lr,
    _save_checkpoint,
    _load_checkpoint
)
logger = logging.getLogger(__name__)

global grad_step, step, epoch, best_f1
grad_step = 0
step = 0
best_f1 = -float('inf')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        required=True, )
    parser.add_argument("--val_data", type=str,
                        required=True, )
    parser.add_argument('--batch_size', type=int, 
                        required=True, default=16)
    parser.add_argument('--epochs', type=int, 
                        required=True, default=10)
    parser.add_argument("--use_fp16", type=bool_flag, 
                        required=True, default=True)
    parser.add_argument("--resume", type=bool_flag, 
                        required=True, default=False)
    parser.add_argument("--seed", type=int, 
                        required=False, default=42)
    parser.add_argument("--grad_acc_steps", type=int,
                        required=True, default=2)
    parser.add_argument("--total_steps", type=int,
                        required=False, default=20000)
    parser.add_argument("--lr", type=float,
                        required=True, default=1e-5)
    parser.add_argument("--lr_schedule", type=str,
                        required=True, default='noam')
    parser.add_argument("--lm_coef", type=float, default=1.0,
                        required=False, help="LM loss coefficient")
    parser.add_argument("--max_seq_len", type=int,
                        required=False, default='128')
    parser.add_argument('--model_type', type=str,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'xlnet'], 
                        default='gpt2-medium')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='for torch distributed')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='location for saving model checkpoints and logs')
    parser.add_argument('--ckpt_file', type=str, required=True,
                        help='pretrained checkpoint file from which training should resume')                
    parser.add_argument('--log_dir', type=str, required=True,
                        help='location for training and validation logs')
    #parser.add_argument('--save_step', type=int, default=1,
    #                    help='number of global steps before saving model')
    #parser.add_argument('--eval_step', type=int, default=1,
    #                    help='number of global steps before evaluating model')

    args = parser.parse_args()
    return args


def run_epoch(model, optimizer, device, config, train_ldr, args):
    model.train()
    global grad_step, step, epoch
    for batch in tqdm(train_ldr, total=860):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_ids, pos_ids, label_ids = batch
        loss, logits, hidden_states = model(input_ids=input_ids,
                                            position_ids=pos_ids,
                                            labels=label_ids)
        if args.n_gpu > 1:
            loss = loss.mean()
        loss *= args.lm_coef
        ppl = torch.exp(loss*input_ids.size()[0] / torch.sum(label_ids != -100))
        loss.backward()
        """
        if args.use_fp16:
            try:
                from apex import amp
                amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError('Apex is not installed ')
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        """
        step += 1     
        if step % args.grad_acc_steps == 0:
            set_lr(optimizer, step, args.lr_schedule,
                   args.lr, tot_steps=args.total_steps,
                   n_embd=config.n_embd)

            optimizer.step()
            optimizer.zero_grad()
            grad_step += 1

            # write to log
            os.makedirs(args.log_dir, exist_ok=True)
            f = open(os.path.join(args.log_dir, 'train_log.txt'),
                        'a', buffering=1)
            print('{}\t{}\t{}\t{:.4f}\t{:.4f}'.format(epoch+1, step, grad_step, 
                                                      loss.item(), ppl.detach().cpu().item()), file=f)


def main(args):
    global grad_step, step, epoch, best_f1
    set_seed(args)

    train_ldr = Loader(args.train_data, args.batch_size,
                       args.max_seq_len, shuffle=True) if args.local_rank == -1 \
                       else DistLoader(get_rank(), get_world_size(), 
                                       args.train_data, args.max_seq_len)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    ##### Load Model and Optimizer #####
    config = AutoConfig.from_pretrained(args.model_type,)
    model = AutoModelWithLMHead.from_pretrained(
        '/home/femi/codebase/model/medium_ft.pkl',
        from_tf=False,
        config=config,
    )
    model.to(args.device)
    optimizer_grouped_parameters = [
    {'params': [p for n, p in list(model.named_parameters())
                if not any(nd in n for nd in ('bias', 'ln'))],
     'weight_decay': 0.01},
    {'params': [p for n, p in list(model.named_parameters())
                if any(nd in n for nd in ('bias', 'ln'))], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.lr, weight_decay=0.01)

    if args.use_fp16:
        logger.info('setting up AMP. training')
        # for fp16 training
        try:
            from apex import amp
            amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Apex is not installed.")
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O1')
    if args.resume:
        _load_checkpoint(args.ckpt_file, model, optimizer, args)

    # multi-gpu training
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    epoch = 0
    for e in range(args.epochs):

        run_epoch(model, optimizer, args.device, config,
                  train_ldr, args)
        ### Evaluate model @@@@@@@@@
        logger.info("Evaluating model on dev set . . .")
        eval_args = {'val_data': args.val_data, 'grad_step': grad_step,
                    'bm_size':1, 'do_sample': True, 'max_resp_len': 1000, 'epoch': epoch,
                    'min_resp_len': 1, 'top_k': 0, 'top_p': 0.9, 'repetition_penalty': 1.2,
                    'temperature': 0.7, 'ngram_sz': 2, 'early_stopping': True,
                    'log_dir': '/home/femi/codebase/logs'}
        f1, rougeL, bleu4, meteor, bleu3, bleu2, bleu1 = eval_model(model, SimpleNamespace(**eval_args), args.device)
        if f1 > best_f1:
            logger.info("saving checkpoint ,{:4f} > {:4f}".format(f1, best_f1))
            _save_checkpoint(model, optimizer, grad_step, args)
            best_f1 = f1
        epoch += 1


if __name__ == '__main__':
    args = setup_args()
    main(args)


"""
python train.py --train_data '/home/femi/codebase/dd_data/train.256len.db' \
                --val_data '/home/femi/codebase/dd_data/val-dirty.txt' \
                --model_type 'gpt2-medium' \
                --batch_size 8 \
                --epochs 100 \
                --use_fp16 False \
                --grad_acc_steps 4 \
                --lr 6.25e-5 \
                --lm_coef 4 \
                --lr_schedule 'noam' \
                --local_rank -1 \
                --resume False \
                --ckpt_file '/home/femi/codebase/model/medium_ft.pkl' \
                --output_dir '/home/femi/codebase/model/' \
                --log_dir '/home/femi/codebase/logs/'


eval_args = {'val_data': '/home/femi/codebase/dd_data/val.txt', 'grad_step': 1,
'bm_size':4, 'do_sample': True, 'max_resp_len': 1000, 'epoch':1,
'min_resp_len': 1, 'top_k': 0, 'top_p': 0.9, 'repetition_penalty': 1.2,
'temperature': 0.7, 'ngram_sz': 2, 'early_stopping': True,
'log_dir': '/home/femi/codebase/logs'}

"""
