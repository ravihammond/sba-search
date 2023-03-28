import os
import sys
import time
import argparse
from collections import defaultdict
import pprint
pprint = pprint.pprint
import numpy as np
import json
import pathlib
from natsort import natsorted, ns
import csv

SPLIT_NAME = { "six": "6-7-splits", "one": "1-12-splits" }


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def combine_games(args):
    for split_type in args.split_type:
        if (len(args.split_type) > 1):
            print(f"\n===== {SPLIT_NAME[split_type]} =====")
        for data_type in args.data_type:
            if (len(args.data_type) > 1):
                print(f"\n--- {data_type.upper()} ---")
            for split_index in args.split_index:
                if (len(args.split_index) > 1):
                    print(f"\n- SPLIT: {split_index} -")
                for model in args.model:
                    print(f"\n=== Model: {model.upper()} ===")
                    combine_for_split(args, split_type, data_type, 
                            split_index, model)

def combine_for_split(args, split_type, data_type, split_index, model):
    splits = load_json_list(f"train_test_splits/sad_splits_{split_type}.json")
    indexes = splits[split_index]["train"]
    indexes = [x + 1 for x in indexes]
    indexes_str = '_'.join(str(x) for x in indexes)
    model_name = f"{model}_sad_{split_type}_{indexes_str}"
    print(f"\n{model_name}\n")

    test_indexes = splits[split_index]["test"]
    for split_index in range(len(test_indexes)):
        partner_index = test_indexes[split_index]
        sad_partner = f"sad_{partner_index + 1}"
        print(f"{sad_partner}")

        combine_for_pair(args, split_type, data_type, split_index, 
                model, splits, model_name, sad_partner, "scores", combine_scores)


def combine_for_pair(args, split_type, data_type, split_index, 
        model, splits, model_name, sad_partner, file_type, combine_files):
    dir_path = os.path.join(
        args.source,
        SPLIT_NAME[split_type],
        data_type,
        model,
        file_type
    )

    if not os.path.exists(dir_path):
        print(bcolors.FAIL + f"No scores found." + bcolors.ENDC)
        return

    file_name = f"{model_name}_vs_{sad_partner}"

    all_file_names = os.listdir(dir_path)
    file_names = [ x for x in all_file_names if sad_partner in x ]
    file_names = natsorted(file_names, alg=ns.IGNORECASE)

    all_file_names_str = " ".join(file_names)
    found_files = []
    missing_files = []

    for seed in args.seeds:
        game_str = f"game_{seed}"
        if game_str in all_file_names_str:
            found_files.append(game_str)
        else:
            missing_files.append(game_str)

    if len(found_files) == len(args.seeds):
        print(bcolors.OKGREEN + f"All {file_type} found, combining." + bcolors.ENDC)
        combine_files(args, file_names, dir_path, file_name)
    elif len(found_files) == 0:
        print(bcolors.FAIL + f"No {file_type} found." + bcolors.ENDC)
    else:
        print(bcolors.WARNING + f"Partial {file_type} found, {len(missing_files)} missing.")
        print(bcolors.ENDC, end="")


def combine_scores(args, file_names, dir_path, model_pair_name):
    scores = []

    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path) as file: score = file.read()
        scores.append(int(score))

    dir_path_obj = pathlib.Path(dir_path)
    truncated_dir_path = os.path.join(*dir_path_obj.parts[1:])
    output_dir_path = os.path.join(args.out, truncated_dir_path)
    output_file_path = os.path.join(output_dir_path, f"{model_pair_name}.csv")

    print(output_file_path)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    scores = [ [x] for x in scores ]
    file = open(output_file_path, 'w+', newline ='')
    with file:
        write = csv.writer(file)
        write.writerows(scores)


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="game_data_combined")
    parser.add_argument("--source", type=str, default="game_data_downloaded")
    parser.add_argument("--model", type=str, default="br")
    parser.add_argument("--split_index", type=str, default="0")
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--seeds", type=str, default="0-100")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    args.model = args.model.split(",")
    args.split_index = [ int(x) for x in args.split_index.split(",") ]
    args.split_type = args.split_type.split(",")
    args.data_type = args.data_type.split(",")

    if '-' in args.seeds:
        seed_range = [ int(x) for x in args.seeds.split('-') ]
        assert(len(seed_range) == 2)
        args.seeds = list(np.arange(*seed_range))
    else:
        args.seeds = [ int(x) for x in args.seeds.split(',') ]

    combine_games(args)

