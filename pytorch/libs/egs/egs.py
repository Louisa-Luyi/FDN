# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29 2020-02-05)

import os
import logging
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io
from libs.support.prefetch_generator import BackgroundGenerator

# There are specaugment and cutout etc..
from .augmentation import *

from collections import defaultdict
from torch.utils.data import Sampler

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer
class ChunkEgs(Dataset):
    """Prepare chunk-egs for time-context-input-nnet (such as xvector which transforms egs form chunk-frames level to
    single-utterance level and TDNN) or single-channel-2D-input-nnet (such as resnet). Do it by linking to egs path
    temporarily and read them in training time, actually.
    The acoustic feature based egs are not [frames, feature-dim] matrix format any more and it should be seen as
    a [feature-dim, frames] tensor after transposing.
    """
    def __init__(self, egs_csv, egs_type="chunk", io_status=True, aug=None, aug_params={}):
        """
        @egs_csv:
            utt-id:str  ark-path:str  start-position:int  end-position:int  class-lable:int

        Other option
        @io_status: if false, do not read data from disk and return zero, which is useful for saving i/o resource
        when kipping seed index.
        """
        assert egs_type is "chunk" or egs_type is "vector"
        assert egs_csv != "" and egs_csv is not None
        head = pd.read_csv(egs_csv, sep=" ", nrows=0).columns

        assert "ark-path" in head
        assert "class-label" in head
        assert "lang" in head

        if egs_type is "chunk":
            if "start-position" in head and "end-position" in head:
                self.chunk_position = pd.read_csv(egs_csv, sep=" ", usecols=["start-position", "end-position"]).values
            elif "start-position" not in head and "end-position" not in head:
                self.chunk_position = None
            else:
                raise TypeError("Expected both start-position and end-position are exist in {}.".format(egs_csv))

        # It is important that using .astype(np.string_) for string object to avoid memeory leak
        # when multi-threads dataloader are used.
        self.ark_path = pd.read_csv(egs_csv, sep=" ", usecols = ["ark-path"]).values.astype(np.string_)
        self.label = pd.read_csv(egs_csv, sep=" ", usecols = ["class-label"]).values
        self.lang = pd.read_csv(egs_csv, sep=" ", usecols = ["lang"]).values
        if "dom" in head:
            self.dom = pd.read_csv(egs_csv, sep=" ", usecols = ["dom"]).values
        else:
            self.dom = None

        self.io_status = io_status
        self.egs_type = egs_type

        # Augmentation.
        self.aug = get_augmentation(aug, aug_params)

    def set_io_status(self, io_status):
        self.io_status = io_status

    def __getitem__(self, index):
        if not self.io_status:
            if self.dom is not None:
                return 0., 0., 0., 0.
            else:
                return 0., 0., 0.

        # Decode string from bytes after using astype(np.string_).
        egs_path = str(self.ark_path[index][0], encoding='utf-8')

        if self.chunk_position is not None:
            chunk = [self.chunk_position[index][0], self.chunk_position[index][1]]
        else:
            chunk = None

        if self.egs_type is "chunk":
            egs = kaldi_io.read_mat(egs_path, chunk=chunk)
        else:
            egs = kaldi_io.read_vec_flt(egs_path)

        target = self.label[index][0]
        lang = self.lang[index][0]
        if self.dom is not None:
            dom = self.dom[index][0]
        # Note, egs which is read from kaldi_io is read-only and
        # use egs = np.require(egs, requirements=['O', 'W']) to make it writeable.
        # It avoids the problem "ValueError: assignment destination is read-only".
        # Note that, do not use inputs.flags.writeable = True when the version of numpy >= 1.17.
        egs = np.require(egs, requirements=['O', 'W'])


        if self.aug is not None:
            return self.aug(egs.T), target, lang, dom
        else:
            if self.dom is not None:
                return egs.T, target, lang, dom
            else:
                return egs.T, target, lang

    def __len__(self):
        return len(self.ark_path)


class BaseBunch():
    """BaseBunch:(trainset,[valid]).
    """
    def __init__(self, trainset, valid=None, use_fast_loader=False, max_prefetch=10,
                 batch_size=512, shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
                 samples_per_speaker=1):

        num_samples = len(trainset)
        num_gpu = 1
        multi_gpu = False
        if utils.use_horovod():
            # Multi-GPU training.
            import horovod.torch as hvd
            # Partition dataset among workers using DistributedSampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                            trainset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=shuffle)
            multi_gpu = True
            num_gpu = hvd.size()
        elif utils.use_ddp():
            # The num_replicas/world_size and rank will be set automatically with DDP.
            if samples_per_speaker > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    KSamplesPerSpeakerBatchSampler(trainset, samples_per_speaker, batch_size))
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    trainset, shuffle=shuffle)

            multi_gpu = True
            num_gpu = dist.get_world_size()
        else:
            train_sampler = None

        if multi_gpu:
            # If use DistributedSampler, the shuffle of DataLoader should be set False.
            shuffle = False

        if use_fast_loader:
            self.train_loader = DataLoaderFast(max_prefetch, trainset, batch_size = batch_size,
                                               shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                                               drop_last=drop_last, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers,
                                           pin_memory=pin_memory, drop_last=drop_last, sampler=train_sampler)

        self.num_batch_train = len(self.train_loader)
        print(self.num_batch_train)

        if self.num_batch_train <= 0:
            raise ValueError("Expected num_batch of trainset > 0. There are your egs info: num_gpu={}, num_samples/gpu={}, "
                             "batch-size={}, drop_last={}.\nNote: If batch-size > num_samples/gpu and drop_last is true, then it "
                             "will get 0 batch.".format(num_gpu, len(trainset)/num_gpu, batch_size, drop_last))


        if valid is not None:
            valid_batch_size = min(batch_size, len(valid)) # To save GPU memory

            if len(valid) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            # Do not use DataLoaderFast for valid for it increases the memory all the time when compute_valid_accuracy is True.
            # But I have not find the real reason.
            self.valid_loader = DataLoader(valid, batch_size = valid_batch_size, shuffle=False, num_workers=num_workers,
                                           pin_memory=pin_memory, drop_last=False)

            self.num_batch_valid = len(self.valid_loader)
        else:
            self.valid_loader = None
            self.num_batch_valid = 0


    @classmethod
    def get_bunch_from_csv(self, trainset_csv:str, valid_csv:str=None, egs_params:dict={}, data_loader_params_dict:dict={}):
        egs_type = "chunk"
        if "egs_type" in egs_params.keys():
            egs_type = egs_params.pop("egs_type")
            if egs_type != "chunk" and egs_type != "vector":
                raise TypeError("Do not support {} egs now. Select one from [chunk, vector].".format(egs_type))

        trainset = ChunkEgs(trainset_csv, **egs_params, egs_type=egs_type)

        # For multi-GPU training.
        if not utils.is_main_training():
            valid = None
        if valid_csv != "" and valid_csv is not None:
            valid = ChunkEgs(valid_csv, egs_type=egs_type)
        else:
            valid = None
        return self(trainset, valid, **data_loader_params_dict)

    def get_train_batch_num(self):
        return self.num_batch_train

    def get_valid_batch_num(self):
        return self.num_batch_valid

    def __len__(self):
        # main: train
        return self.num_batch_train

    @classmethod
    def get_bunch_from_egsdir(self, egsdir:str, egs_params:dict={}, data_loader_params_dict:dict={}):
        train_csv_name = None
        valid_csv_name = None

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        feat_dim, num_targets, train_csv, valid_csv = get_info_from_egsdir(egsdir,
                                                         train_csv_name=train_csv_name, valid_csv_name=valid_csv_name)
        info = {"feat_dim":feat_dim, "num_targets":num_targets}
        bunch = self.get_bunch_from_csv(train_csv, valid_csv, egs_params, data_loader_params_dict)
        return bunch, info


class KSamplesPerSpeakerBatchSampler(Sampler):
    def __init__(self, dataset, k, batch_size):
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size

        labels = dataset.label.reshape(-1)  # Flatten the label array
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[label_counts >= k]  # Filter labels with at least k samples
        label_to_indices = {label: np.where(labels == label)[0] for label in valid_labels}

        self.indices = []
        for label, indices in label_to_indices.items():
            num_samples = len(indices)
            num_batches = num_samples // k
            num_samples = num_batches * k
            indices = indices[:num_samples]
            indices = indices.reshape(-1, k)
            self.indices.extend(indices.tolist())

        self.num_batches = len(self.indices) // batch_size

    def __iter__(self):
        np.random.shuffle(self.indices)
        batch_indices = self.indices[: self.num_batches * self.batch_size]
        batch_indices = np.array(batch_indices)
        batch_indices = batch_indices.reshape(-1, self.batch_size)
        return iter(batch_indices.tolist())

    def __len__(self):
        return self.num_batches


def create_data_loader_with_k_samples_per_speaker(trainset, valid=None, k=1, **data_loader_params_dict):
    train_sampler = KSamplesPerSpeakerBatchSampler(trainset, k=k)
    if "batch_size" in data_loader_params_dict:
        del data_loader_params_dict["batch_size"]
    data_loader_params_dict.update({"batch_sampler": train_sampler})
    return BaseBunch(trainset, valid, **data_loader_params_dict)


class DataLoaderFast(DataLoader):
    """Use prefetch_generator to fetch batch to avoid waitting.
    """
    def __init__(self, max_prefetch, *args, **kwargs):
        assert max_prefetch >= 1
        self.max_prefetch = max_prefetch
        super(DataLoaderFast, self).__init__(*args, **kwargs)

    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderFast, self).__iter__(), self.max_prefetch)

## Function
def get_info_from_egsdir(egsdir, train_csv_name=None, valid_csv_name=None):
    if os.path.exists(egsdir+"/info"):
        feat_dim = int(utils.read_file_to_list(egsdir+"/info/feat_dim")[0])
        num_targets = int(utils.read_file_to_list(egsdir+"/info/num_targets")[0])

        train_csv_name = train_csv_name if train_csv_name is not None else "train.egs.csv"
        valid_csv_name = valid_csv_name if valid_csv_name is not None else "valid.egs.csv"

        train_csv = egsdir + "/" + train_csv_name
        valid_csv = egsdir + "/" + valid_csv_name

        if not os.path.exists(valid_csv):
            valid_csv = None

        return feat_dim, num_targets, train_csv, valid_csv
    else:
        raise ValueError("Expected dir {0} to exist.".format(egsdir+"/info"))
