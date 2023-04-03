import os
import sys
import time
import argparse
from collections import defaultdict
import pprint
pprint = pprint.pprint
import numpy as np
from google.cloud import storage
import json

PROJECT = "aiml-reid-research"
GCLOUD_PATH = "Ravi"
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


def check_gcloud(args):
    print(f"checking: {args.dir}\n")
    client = storage.Client(project=PROJECT)
    bucket = client.get_bucket(PROJECT + "-data")

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
                    check_files(args, client, bucket, split_type, 
                            data_type, split_index, model)


def check_files(args, client, bucket, split_type, data_type, split_index, model):
    splits = load_json_list(f"train_test_splits/sad_splits_{split_type}.json")
    indexes = splits[split_index]["train"]
    indexes = [x + 1 for x in indexes]
    indexes_str = '_'.join(str(x) for x in indexes)
    model_name = f"{model}_sad_{split_type}_{indexes_str}"
    print(f"\n{model_name}\n")

    prefix = os.path.join(
        GCLOUD_PATH, 
        args.dir,
        SPLIT_NAME[split_type],
        data_type,
        model,
        "games",
        model_name
    )
    all_blobs = list(client.list_blobs(bucket, prefix=prefix))

    if (len(all_blobs) == 0):
        print(bcolors.FAIL + "No games found." + bcolors.ENDC)
        return

    blobs = defaultdict(list)
    filepaths = []

    for blob in all_blobs:
        filepaths.append(blob.name)

    test_indexes = splits[split_index]["test"]

    for split_index in range(len(test_indexes)):
        partner_index = test_indexes[split_index]
        sad_partner = f"sad_{partner_index + 1}"
        print(f"{sad_partner}")

        found_list = [ x for x in filepaths if sad_partner in x]
        found_list_str = "\t".join(found_list)
        missing_games = []
        found_games = []

        for seed in args.seeds:
            game_str = f"game_{seed}"
            if game_str not in found_list_str:
                missing_games.append(game_str)
            else:
                found_games.append(game_str)

        if len(missing_games) == 0:
            print(bcolors.OKGREEN + "All games finished." + bcolors.ENDC)
        elif len(missing_games) == len(args.seeds):
            print(bcolors.FAIL + "No games finished." + bcolors.ENDC)
        else:
            print(bcolors.WARNING + f"Partial games finished, {len(missing_games)} games left.")
            if args.verbose:
                for game_str in found_games:
                    print(bcolors.OKGREEN + game_str + bcolors.ENDC)
                for game_str in missing_games:
                    print(bcolors.FAIL + game_str + bcolors.ENDC)
            print(bcolors.ENDC, end="")


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="br,sba")
    parser.add_argument("--split_index", type=str, default="0")
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--seeds", type=str, default="0-100")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--dir", type=str, default="hanabi-search-games-br")
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

    check_gcloud(args)

