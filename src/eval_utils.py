import argparse
import datasets  # HF's datasets package
import logging
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer
)
from metrics.pycocoevalcap.f1 import f1_score  
from torch.nn.utils.rnn import pad_sequence
from utils import bool_flag                   

logger = logging.getLogger(__name__)

EOS = "<|endoftext|>"
toker = AutoTokenizer.from_pretrained('gpt2-medium')
f1 = datasets.load_metric('f1')
meteor = datasets.load_metric('meteor')
rouge = datasets.load_metric('rouge')
bleu = datasets.load_metric('sacrebleu')

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data", type=str,
                        required=True, )
    parser.add_argument("--ckpt_file", type=str,
                        required=True, )
    parser.add_argument('--bm_size', type=int,
                        default=True, required=True)
    parser.add_argument('--do_sample', type=bool_flag,
                        default=True, required=True)
    parser.add_argument('--min_resp_len', type=int,
                        default=1, required=True)
    parser.add_argument('--max_resp_len', type=int,
                        default=100, required=True)
    parser.add_argument('--top_k', type=int,
                        default=40, required=False)
    parser.add_argument('--top_p', type=float,
                        default=0.9, required=False)
    parser.add_argument('--temperature', type=float,
                        default=0.8, required=False)
    parser.add_argument('--ngram_sz', type=int,
                        default=2, required=True)
    parser.add_argument('--repetition_penalty', type=float,
                        default=1.2, required=True)
    parser.add_argument('--epoch', type=int,
                        default=1, required=True)
    parser.add_argument('--grad_step', type=int,
                        default=2000, required=True)
    parser.add_argument('--early_stopping', type=bool_flag,
                        default=True, required=True)
    parser.add_argument('--log_dir', type=str, required=True,
                        help='location for validation logs')
    args = parser.parse_args()
    return args


def preprocess(line: str) -> str:
    """ clean line instance """
    # mysterious_toks = ['¥', '—', '°','′', '£', '‘', '、', '。', '“', '”']
    line = line.replace('—', "-")
    line = line.replace('‘', "'").replace('′', "'").replace('“', '/"').replace('”', '/"').replace('、', ', ')
    line = line.replace('° C', ' degree Celsius')
    line = line.replace('。', '.')
    # preprocess("I need to remit £ 500 to London. 20° C, ¥ 45 $ 100")
    curr = {"¥": ' yuan', "$": "dollars", "£": "euros"}
    for symbol in curr.keys():
        if symbol in line:
            idx1 = line.find(symbol)
            line = line.replace(line[idx1], curr[symbol])
            #print(symbol, '\t', line)
            #exit()
    return line


def eval_model(model, args, device):
    model = model.module if hasattr(model, 'module') else model  # remove dataparallel wrapper
    model.to(device)
    model.eval()
    tot_loss, tot_ppl, tot_f1, tot_bleu4, tot_bleu3, tot_bleu2, tot_bleu1, tot_meteor, tot_rougeL = \
    [], [], [], [], [], [], [], [], []

    with torch.no_grad():
        logger.info("preprocessing data . . .\n")
        lines = open(args.val_data).readlines()
        lines = list(map(preprocess, lines))
        logger.info("evaluating . . .\n")
        for line in tqdm(lines):

            turns = [x.strip() for x in line.rstrip().split('__eou__') if x]
            ctxt, label = ' <|endoftext|> '.join(turns[:-1]), turns[-1]
            input_id = toker.encode(ctxt+' <|endoftext|>', return_tensors='pt')
            label_id = toker.encode(label, return_tensors='pt')
            padded = pad_sequence([torch.t(input_id), torch.t(label_id)], padding_value=0)
            loss, _, _ = model(input_ids=torch.t(padded[:, 0, :]).to(device),
                               labels=torch.t(padded[:, 1, :]).to(device))

            ppl = torch.exp(loss.cpu()/label_id.shape[-1]).detach().cpu().item()

            output_ids = model.generate(input_id.to(device),
                                        pad_token_id=50256,
                                        eos_token_id=50256,
                                        do_sample=args.do_sample,
                                        num_beams=args.bm_size,
                                        max_length=args.max_resp_len,
                                        min_length=args.min_resp_len,
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        temperature=args.temperature,
                                        repetition_penalty=args.repetition_penalty,
                                        no_repeat_ngram_size=args.ngram_sz,
                                        early_stopping=args.early_stopping)

            resp = output_ids.cpu()[0][input_id.shape[1]:]
                
            gen_resp = toker.decode(resp, skip_special_tokens=True)

            f1 = f1_score(gen_resp, label)
            met = meteor.compute(predictions=[gen_resp], references=[label])['meteor']
            rougeL = rouge.compute(predictions=[gen_resp], references=[label])['rougeL']
            bleu1, bleu2, bleu3, bleu4 = bleu.compute(predictions=[gen_resp], references=[[label]])['precisions']
            tot_loss.append(loss.cpu().item())
            tot_ppl.append(ppl)
            tot_f1.append(f1)
            tot_bleu4.append(bleu4)
            tot_bleu3.append(bleu3)
            tot_bleu2.append(bleu2)
            tot_bleu1.append(bleu1)
            tot_meteor.append(met)
            tot_rougeL.append(rougeL.mid.fmeasure)
    
    log_file = os.path.join(args.log_dir, 'eval_log.txt')
    num_examples = len(tot_f1)

    with open(log_file, 'a') as f:
        f.write('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
        args.epoch, args.grad_step, sum(tot_ppl)/num_examples, sum(tot_f1)/num_examples*100, 
        sum(tot_rougeL)/num_examples*100, sum(tot_bleu4)/num_examples, sum(tot_meteor)/num_examples,
        sum(tot_bleu2)/num_examples, sum(tot_bleu3)/num_examples, sum(tot_bleu1)/num_examples))

    return (sum(tot_f1)/num_examples*100,
            sum(tot_rougeL)/num_examples*100,
            sum(tot_bleu4)/num_examples,
            sum(tot_meteor)/num_examples,
            sum(tot_bleu3)/num_examples,
            sum(tot_bleu2)/num_examples,
            sum(tot_bleu1)/num_examples)


def main():
    args = setup_args()
    ckpt = torch.load(args.ckpt_file)
    config = AutoConfig.from_pretrained('gpt2-medium')
    model = AutoModelWithLMHead.from_pretrained(
        'gpt2-medium',
        from_tf=False,
        config=config,
    )
    model.load_state_dict(ckpt['model_state'])
    device = torch.device('cuda:1')
    eval_model(args.model, args, device)

if __name__ == '__main__':
    main()
"""
python eval_utils.py --val_data /home/femi/codebase/dd_data/val.txt \
                     --ckpt_file /home/femi/codebase/model/GP2-pretrain-step-5101.pkl \
                     --bm_size 1 \
                     --do_sample True \
                     --max_resp_len 1000 \
                     --min_resp_len 1 \
                     --top_k 40 \
                     --top_p 0.9 \
                     --repetition_penalty 1.2 \
                     --temperature 0.6 \
                     --ngram_sz 2 \
                     --early_stopping True \
                     --log_dir /home/femi/codebase/logs \
                     --epoch 1 \
                     --grad_step 2000

"""
