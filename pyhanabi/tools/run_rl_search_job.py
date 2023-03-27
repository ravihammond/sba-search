import os
import sys
import argparse
import pprint
pprint = pprint.pprint
import numpy as np
import json
from easydict import EasyDict as edict
import copy
from multiprocessing import Pool
import subprocess


def run_rl_search_jobs(args):
    jobs = create_search_jobs(args)
    run_jobs(args, jobs)

def create_search_jobs(args):
    game_seeds = get_game_seeds(args.seeds)
    player_weight = model_to_weight(args, args.model)
    partner_weight = model_to_weight(args, args.partner_model)

    jobs = []
    job = edict()
    job.player, job.parnter = None, None
    job.sad_legacy = [0, 0]
    job.name = ["", ""]

    job.player, job.sad_legacy[1], job.name[1] = model_to_weight(args, args.model)
    job.partner, job.sad_legacy[0], job.name[0] = model_to_weight(args, args.partner_model)
    split_type_map = {"one": "1-12-splits", "six": "6-7-splits"}
    split_type_str = split_type_map[args.split_type]
    job.save_dir = f"game_data/{split_type_str}/{args.data_type}/{args.model}"

    for i, game_seed in enumerate(game_seeds):
        new_job = copy.deepcopy(job)
        new_job.game_seed = game_seed
        new_job.device1 = f"cuda:{(i % 2) * 2}"
        new_job.device2 = f"cuda:{((i % 2) * 2) + 1}"
        jobs.append(new_job)

    return jobs


def get_game_seeds(game_seeds):
    if '-' in game_seeds:
        seed_range = [int(x) for x in game_seeds.split('-')]
        assert(len(seed_range) == 2)
        game_seed_list = list(np.arange(*seed_range))
    else:
        game_seed_list = [int(x) for x in game_seeds.split(',')]

    return game_seed_list


def model_to_weight(args, model):
    splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
    indexes = splits[args.split_index]["train"]
    indexes = [x + 1 for x in indexes]
    indexes_str = '_'.join(str(x) for x in indexes)

    player_name = model

    if model == "br":
        player_name = f"br_sad_{args.split_type}_{indexes_str}"
        path = f"../models/my_models/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif model == "sba":
        player_name = f"sba_sad_{args.split_type}_{indexes_str}"
        path = f"../models/my_models/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif "obl" in model:
        policies = load_json_list("agent_groups/all_obl.json")
        path = policies[args.partner_index]
        player_name = f"obl_{args.partner_index + 1}"
        sad_legacy = 0

    elif model == "op":
        policies = load_json_list("agent_groups/all_op.json")
        path = policies[args.partner_index]
        player_name = f"op_{args.partner_index + 1}"
        sad_legacy = 1

    elif model == "sad":
        policies = load_json_list("agent_groups/all_sad.json")
        sad_index = splits[args.split_index][args.data_type][args.partner_index]
        path = policies[sad_index]
        player_name = f"sad_{sad_index + 1}"
        sad_legacy = 1

    return path, sad_legacy, player_name


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def run_jobs(args, jobs):
    with Pool(processes=args.workers) as pool:
        results = pool.map(run_job, jobs)


def run_job(job):
    command = ["python", "rl_search.py",
        "--save_dir", job.save_dir,
        "--weight1", job.partner,
        "--weight2", job.player,
        "--sad_legacy", ",".join([str(x) for x in job.sad_legacy]),
        "--player_name", ",".join(job.name),
        "--data_type", args.data_type,
        "--split_type", args.split_type,
        "--game_seed", str(job.game_seed),
        "--seed", str(job.game_seed),
        "--burn_in_frames", "5000",
        "--replay_buffer_size", "100000",
        "--rl_rollout_device", job.device2,
        "--bp_rollout_device", job.device2,
        "--train_device", job.device1,
        "--belief_device", job.device1,
        "--rollout_batchsize", "8000",
        "--num_thread", "1",
        "--batchsize", "128",
        "--num_epoch", "1",
        "--epoch_len", "5000",
        "--num_samples", "50000",
        "--skip_search", "0",
        "--ad_hoc", "1",
        "--upload_gcloud", "1",
        "--save_game", "1",
        "--verbose", "1"
    ]

    subprocess.run(command)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="br")
    parser.add_argument("--partner_model", type=str, default="sad")
    parser.add_argument("--partner_index", type=int, default=0)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip_search", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_rl_search_jobs(args)

