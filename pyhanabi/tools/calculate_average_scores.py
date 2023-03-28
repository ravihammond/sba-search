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


def calculate_all_average_scores(args):
    for model in args.model:
        print(f"\n=== Model: {model.upper()} ===")
        for split_type in args.split_type:
            if (len(args.split_type) > 1):
                print(f"\n===== {SPLIT_NAME[split_type]} =====")
            for data_type in args.data_type:
                if (len(args.data_type) > 1):
                    print(f"\n--- {data_type.upper()} ---")

                calculate_average_scores(args, model, split_type, data_type)


def calculate_average_scores(args, model, split_type, data_type):
    print()
    dir_path = os.path.join(
        args.dir,
        SPLIT_NAME[split_type],
        data_type,
        model,
        "scores"
    )

    if not os.path.exists(dir_path):
        print(bcolors.FAIL + f"Path does not exist." + bcolors.ENDC)
        return 

    splits = load_json_list(f"train_test_splits/sad_splits_{split_type}.json")

    all_scores = []

    for split_index in args.split_index:
        if (len(args.split_index) > 1):
            print(f"\n- SPLIT: {split_index} -")

        split_scores = []

        indexes = splits[split_index]["train"]
        indexes = [x + 1 for x in indexes]
        indexes_str = '_'.join(str(x) for x in indexes)
        model_name = f"{model}_sad_{split_type}_{indexes_str}"

        test_indexes = splits[split_index]["test"]
        for partner_index in range(len(test_indexes)):
            partner_num = test_indexes[partner_index]
            partner_name = f"sad_{partner_num + 1}"
            pair_str = f"{model_name}_vs_{partner_name}"
            file_path = os.path.join(dir_path, f"{pair_str}.csv")

            if not os.path.exists(file_path):
                print(bcolors.FAIL + f"Path {file_path} not found." + bcolors.ENDC)
                continue

            scores = []

            with open(file_path, newline='') as file:
                reader = csv.reader(file)
                scores = [int(x[0]) for x in list(reader)]

            if len(scores) == 0:
                print(bcolors.FAIL + f"Csv is empty." + bcolors.ENDC)
                continue 

            # print(pair_str)
            print(partner_name)
            mean = np.mean(scores)
            sem = np.std(scores) / np.sqrt(len(scores))
            print(f"{mean:.3f} ± {sem:.3f}")

            split_scores = [*split_scores, *scores]

        print(f"\n{model_name}")
        mean = np.mean(split_scores)
        sem = np.std(split_scores) / np.sqrt(len(split_scores))
        print(f"{mean:.3f} ± {sem:.3f}")

        all_scores = [*all_scores, *split_scores]

    print(f"\n{model}")
    mean = np.mean(all_scores)
    sem = np.std(all_scores) / np.sqrt(len(all_scores))
    print(f"{mean:.3f} ± {sem:.3f}")
            

def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="game_data_combined")
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

    calculate_all_average_scores(args)

