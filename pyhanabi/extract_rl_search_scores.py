import argparse
import os
import csv
import pprint
pprint = pprint.pprint
import re
import numpy as np

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def get_last_line(filename):
    with open(filename, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line

def extract_scores(args):
    models = args.models.split(",")
    all_dirs = os.listdir(args.path)
    filtered_models = [
        dir 
        for dir in all_dirs 
        for model in models
        if model in dir
    ]

    for model_name in filtered_models:
        model_path = os.path.join(args.path, model_name)
        runs = sort_human(os.listdir(model_path))
        scores = []

        for run in runs:
            run_path = os.path.join(model_path, run)
            last_line = get_last_line(run_path).strip()

            if "Final Score" not in last_line:
                if args.verbose:
                    print(f"Run {run} incomplete.")
                continue

            final_score = int(re.search(r'\d+', last_line).group())

            if args.partner is None:
                scores.append(final_score)
            elif args.partner in run:
                scores.append(final_score)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        save_path = os.path.join(args.out_dir, model_name + ".csv")
        np.savetxt(save_path, scores, delimiter=",",fmt='%1.4f')

        # if args.max_samples
        # score = score[:]
        print("model_name:", model_name)
        mean = np.mean(scores)
        stderrmean = np.std(scores) / np.sqrt(len(scores))
        print(f"avg score: {mean:.2f} ± {stderrmean:.2f}")
        print(f"num samples: {len(scores)}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--path", type=str, default="rl_data")
    parser.add_argument("--partner", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="rl_data_scores")
    args = parser.parse_args()

    extract_scores(args)

