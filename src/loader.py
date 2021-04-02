import dbm
import gzip
import json
import math
import random
import torch

from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence


class DialogDataset(Dataset):
    def __init__(self, dlg_fts, max_len):
        self.dlg_fts = dlg_fts
        self.max_len = max_len
        # do this

    def __getitem__(self, i):
        _feat = self.dlg_fts[i]
        input_ids = _feat['input_ids']
        label_ids = _feat["label_ids"]
        position_ids = _feat["position_ids"]
        token_type_ids = _feat["token_type_ids"]
        # remove long dialogs
        if len(input_ids) > self.max_len:
            input_ids, label_ids, position_ids, token_type_ids = input_ids[-max_len:], \
                                                                 label_ids[-max_len:], \
                                                                 position_ids[-max_len:], \
                                                                 token_type_ids[-max_len:]
        return {'input_ids': input_ids,
                'label_ids': label_ids,
                'position_ids': position_ids,
                'token_type_ids': token_type_ids
                }

    def __len__(self):
        return len(self.dlg_fts)

    def collate(features):
        input_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long)
                                 for f in features], batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f['position_ids'], dtype=torch.long) for f in features],
                                    batch_first=True,
                                    padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f['token_type_ids'], dtype=torch.long) for f in features],
                                      batch_first=True,
                                      padding_value=0)
        labels = pad_sequence([torch.tensor(f['label_ids'], dtype=torch.long) for f in features],
                              batch_first=True,
                              padding_value=-100)
        return (input_ids,
                position_ids,
                token_type_ids,
                labels)


class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class Loader(DataLoader):
    """ this loads shelve db chunks and then convert to mini-batch loader"""
    def __init__(self, db_name, batch_size, max_len,
                 bucket_size=100, shuffle=True):
        self.db = dbm.open(f'{db_name}', 'r')
        self.batch_size = batch_size
        self.max_len = max_len
        self.bucket_size = bucket_size * batch_size
        self.shuffle = shuffle

    def _get_keys(self):
        keys = list(self.db.keys())
        return keys

    def __iter__(self):
        keys = self._get_keys()
        if self.shuffle:
            random.shuffle(keys)
        for key in keys:
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))
            # discard long examples
            trunc_chunk = []
            lens = []
            for feat in chunk:
                if len(feat['input_ids']) > self.max_len:
                    continue
                trunc_chunk.append(feat)
                lens.append(len(feat['input_ids']))
            dataset = DialogDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                    droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=8,  # can test multi-worker
                                collate_fn=DialogDataset.collate)
            yield from loader

    def __len__(self):
        raise NotImplementedError()

    def __del__(self):
        self.db.close()


class DistLoader(Loader):
    """ distributed version """
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica

    def _get_keys(self):
        keys = list(self.db.keys())[self.rank::self.num_replica]
        return keys


class BatchLoader():
    def __init__(self, corpus_file, batch_size, max_len):
        self.corpus = corpus_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_examples = len(open(self.corpus, 'r').readlines())
    
    
    def __len__(self):
        return math.ceil(self.num_examples / self.batch_size)