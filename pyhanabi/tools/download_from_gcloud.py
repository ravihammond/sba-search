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
from google.cloud import storage

PROJECT = "aiml-reid-research"
GCLOUD_PATH = "Ravi/hanabi-search-games-sba/"
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


def download_from_gcloud(args):
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
                    download_all_files_for_split(args, client, bucket, 
                            split_type, data_type, split_index, model)


def download_all_files_for_split(args, client, 
        bucket, split_type, data_type, split_index, model):
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

        download_all_files(args, client, bucket, split_type, 
                data_type, split_index, model, splits, model_name, 
                sad_partner, "games", "pkl")
        download_all_files(args, client, bucket, split_type, 
                data_type, split_index, model, splits, model_name, 
                sad_partner, "scores", "txt")
        download_all_files(args, client, bucket, split_type, 
                data_type, split_index, model, splits, model_name, 
                sad_partner, "logs", "log")

def download_all_files(args, client, bucket, split_type, data_type, 
            split_index, model, splits, model_name, 
            sad_partner, file_type, ext):
    prefix = os.path.join(
        GCLOUD_PATH, 
        SPLIT_NAME[split_type],
        data_type,
        model,
        file_type,
        f"{model_name}_vs_{sad_partner}"
    )
    all_blobs = list(client.list_blobs(bucket, prefix=prefix))

    if (len(all_blobs) == 0):
        print(bcolors.FAIL + f"No {file_type} found." + bcolors.ENDC)
        return

    blobs = defaultdict(list)
    blobs = []

    for blob in all_blobs:
        blobs.append(blob)

    test_indexes = splits[split_index]["test"]
    partner_index = test_indexes[split_index]
    sad_partner = f"sad_{partner_index + 1}"

    found_list_str = "\t".join(x.name for x in blobs)
    found_blobs = []
    missing_blobs = []

    for seed in args.seeds:
        game_str = f"game_{seed}"
        if game_str in found_list_str:
            found_blobs.append(game_str)
        else:
            missing_blobs.append(game_str)

    if len(found_blobs) == len(args.seeds):
        print(bcolors.OKGREEN + f"All {file_type} finished, downloading." + bcolors.ENDC)
        download_bloblist(args, client, bucket, blobs)
    elif len(found_blobs) == 0:
        print(bcolors.FAIL + f"No {file_type} finished." + bcolors.ENDC)
    else:
        print(bcolors.WARNING + f"Partial {file_type} finished, {len(missing_blobs)} left, not downloading.")
        print(bcolors.ENDC, end="")


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def download_bloblist(args, client, bucket, blobs):
    for blob in blobs:
        blob_path_obj = pathlib.Path(blob.name)
        truncated_blob_path = os.path.join(*blob_path_obj.parts[2:])
        output_path = os.path.join(args.out, truncated_blob_path)

        output_dir_path = os.path.dirname(output_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        blob.download_to_filename(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="game_data_downloaded_sba")
    parser.add_argument("--model", type=str, default="sba")
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

    download_from_gcloud(args)

