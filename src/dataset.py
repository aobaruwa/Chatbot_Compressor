""" Inherits from the logic in https://github.com/microsoft/DialoGPT/blob/master/prepro.py """

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List
import argparse
import dbm
import gzip
import json
import logging
import os

logger = logging.getLogger(__name__)

EOS = "<|endoftext|>"
Tokenizer = AutoTokenizer.from_pretrained('gpt2-medium', use_fast=True)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True,
                        help="directory of the corpus files")
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='discard data longer than this')
    parser.add_argument('--tokenizer', type=str, default="gpt2",
                        help='tokenizer to use')
    args = parser.parse_args()
    return args


def preprocess(line: str) -> str:
    """ clean line instance
    lines like: "I need to remit £ 500 to London at 20° C, converted from ¥ 45 and $ 100"
    """
    # oov_toks = ['¥', '—', '°','′', '£', '‘', '、', '。', '“', '”']
    curr = {"¥": "yuan", "£": "euro"}
    line = line.replace('—', "-")
    line = line.replace('‘', "'").replace('′', "'").replace('“', '/"').replace('”', '/"').replace('、', ', ')
    line = line.replace('° C', ' degree Celsius')
    line = line.replace('。', '.')
    # rectify prices
    if len(set(curr.keys()).intersection(set(line))) == 0:  # no price in line
        return line
    for symbol in curr.keys():
        line = line.replace(symbol+' ', symbol)
    prices = list(filter(is_price, line.split()))
    rectified_prices = {price: price_to_text(price) for price in prices}
    for price in prices:
        line = line.replace(price, rectified_prices[price])
    return line


def is_price(token: str) -> str:
    curr = {"¥": "yuan", "£": "euro"}
    price_curr = token[0]
    return price_curr in curr.keys()


def price_to_text(money: str) -> str:
    """ prices like ¥400.23 become '400.23 yuan' """
    curr = {"¥": "yuan", "£": "euro"}
    return f"{money[1:]} {curr[money[0]]}"


def read_file(args):
    file_name = args.corpus_dir + "train.txt"

    db_path = f'{args.corpus_dir}train.{args.max_seq_len}len.db'
    print(db_path)
    if os.path.exists(db_path):
        raise ValueError("Existing db file {} found!".format(db_path))
    with dbm.open(db_path, 'n') as db:
        chunk = []
        n_chunk = 0
        n_examples = 0
        logger.info("preprocessing data . . .")
        reader = open(file_name, "r", encoding="utf-8")
        # preprocess lines first
        lines = list(map(preprocess, reader.readlines()))
        for line in tqdm(lines):
            try:
                if len(chunk) >= args.chunk_size:
                    # save and renew chunk
                    db[f'chunk_{n_chunk}'] = gzip.compress(
                        json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
                    chunk = chunk[args.chunk_size:]
                    n_chunk += 1
                conv = line.split('__eou__')[:-1]
                dlg = Dialog(conv[:-2], conv[-1], args)
                features = dlg.featurize()
                if len(features["input_ids"]) > args.max_seq_len:
                    continue
                chunk.append(features)
                n_examples += 1
            except Exception as e:
                print('!!! prepro exception !!!', e)
                print(line)
                exit()
        # save last chunk
        db[f'chunk_{n_chunk}'] = gzip.compress(json.dumps(chunk).encode('utf-8'))
    # save relevant information to reproduce
    meta = {'n_example': n_examples,
            'chunk_size': args.chunk_size,
            'max_seq_len': args.max_seq_len
            }
    with open(os.path.join(os.path.dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    return


class Dialog():
    def __init__(self, context: List[str], response: str, args):
        self.context = context
        self.response = response
        self.n_turns = len(context)
        self.args = args

    def featurize(self):
        input_conv = '<|endoftext|>'.join(turn.rstrip()
                                          for turn in self.context)
        ctxt_ids = Tokenizer.encode(input_conv)
        resp_ids = Tokenizer.encode(self.response)
        EOS_TOK = Tokenizer.encode(EOS)
        input_ids = ctxt_ids + EOS_TOK + resp_ids + EOS_TOK  # last eos prompts model for response 

        token_type_ids = [0]*(len(ctxt_ids)+1) + [1]*(len(resp_ids)+1)

        position_ids = list(range(len(input_ids)))

        label_ids = [-100]*len(ctxt_ids) + resp_ids + EOS_TOK + [-100]  # [-100] pads label_ids to shape of input_ids

        assert len(input_ids) == len(token_type_ids) \
            == len(position_ids) == len(label_ids), "lengths of the sequence \
                                                     ids must match"
        feat = {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "label_ids": label_ids}
        return feat

    def __repr__(self):
        dlg = "--EOT-- ".join(self.context) + '\t' + self.response
        return dlg

    def __str__(self):
        return self.__repr__()


def main(args):
    read_file(args)


if __name__ == '__main__':
    args = setup_args()
    main(args)
