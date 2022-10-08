import os
import json
import time
import numpy as np
import torch

from utils.misc import set_random_seed
from utils.distributed import init_distributed, is_default_gpu

from models.vlnbert_init import get_tokenizer

from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch
from r2r.parser import parse_args


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    if args.dataset == 'r2r_back':
        dataset_class = R2RBackBatch
    else:
        dataset_class = R2RBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
        sel_data_idxs=None, name='train'
    )
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name='aug'
        )
    else:
        aug_env = None

    # val_env_names = ['val_train_seen', 'val_seen']
    # if args.test or args.dataset != 'r4r':
    #     val_env_names.append('val_unseen')
    # else:  # val_unseen of r4r is too large to evaluate in training
    #     val_env_names.append('val_unseen_sampled')
    #
    # if args.submit:
    #     if args.dataset == 'r2r':
    #         val_env_names.append('test')
    #     elif args.dataset == 'rxr':
    #         val_env_names.extend(['test_challenge_public', 'test_standard_public'])

    val_envs = {}
    # for split in val_env_names:
    #     val_instr_data = construct_instrs(
    #         args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
    #     )
    #     val_env = dataset_class(
    #         feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
    #         angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
    #         sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split
    #     )
    #     val_envs[split] = val_env

    return train_env, val_envs, aug_env


def pretty_json_dump(obj, fp):
  json.dump(obj, fp, sort_keys=True, indent=2, separators=(',', ':'))


def extract_vision_task(env):
    vision_task = []
    vision_id = 100000
    for scan, shortest_paths in env.shortest_paths.items():
        for start, start_paths in shortest_paths.items():
            for end, path in start_paths.items():
                if len(path) >= 4 and len(path) <= 7:
                    vision_task.append({
                        'distance': env.shortest_distances[scan][start][end],
                        'scan': scan,
                        'path_id': vision_id,
                        'path': path,
                        'heading': 0,
                    })
                    vision_id += 1
    print("vision_task numbers : ", len(vision_task))
    with open("../datasets/R2R/annotations/R2R_vision_train.json", "w") as f:
        pretty_json_dump(vision_task, f)


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)
    extract_vision_task(train_env)


if __name__ == '__main__':
    main()
