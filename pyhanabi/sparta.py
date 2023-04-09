# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import pprint
import torch
import numpy as np

# c++ backend
import set_path

set_path.append_sys_path()
import rela
import hanalearn
import utils
import common_utils


def run_search(args):
    torch.backends.cudnn.deterministic = True
    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)

    bp, config = load_model(args.weight, args.sad_legacy, args.device)
    bp.train(False)

    bp_runner = rela.BatchRunner(bp, args.device, 2000, ["act"])
    bp_runner.start()

    seed = args.seed
    actors = []
    for i in range(args.num_player):
        actor = hanalearn.SpartaActor(i, bp_runner, seed, [args.sad_legacy], None)
        seed += 1
        actors.append(actor)

    actors[args.search_player].set_partners(actors)

    moves, score = run(
        args.game_seed,
        actors,
        args.search_player,
        args.num_search,
        args.threshold,
        args.num_thread,
    )


def load_model(weight, sad_legacy, device):
    if sad_legacy:
        model = utils.load_sad_model(
                weight, 
                device,
                multi_step=1)
        config = {
            "sad": True,
            "hide_action": False,
            "weight": weight,
            "parameterized": False,
            "sad_legacy": True,
            "multi_step": 1,
            "boltzmann_act": False,
            "method": "iql",
        }
    else:
        if "fc_v.weight" in torch.load(weight).keys():
            model, config = utils.load_agent(weight, {"device": device})
            assert not config["hide_action"]
            assert not config["boltzmann_act"]
        else:
            model = utils.load_supervised_agent(weight, device)
            config = {}

    return model, config


def run(seed, actors, search_actor_idx, num_search, threshold, num_thread):
    params = {
        "players": str(len(actors)),
        "seed": str(seed),
        "bomb": str(0),
        "hand_size": str(5),
        "random_start_player": str(0),  # do not randomize start_player
    }
    game = hanalearn.GameSimulator(params)
    step = 0
    moves = []
    while not game.terminal():
        print("\n================STEP %d================\n" % step)
        print(game.state().to_string())

        cur_player = game.state().cur_player()

        actors[search_actor_idx].update_belief(game, num_thread)
        for i, actor in enumerate(actors):
            print(f"\n---Actor {i} observe---")
            actor.observe(game)

        for i, actor in enumerate(actors):
            print(f"\n---Actor {i} decide action---")
            action = actor.decide_action(game)
            if i == cur_player:
                move = game.get_move(action)

        # run sparta, this may change the move
        if cur_player == search_actor_idx:
            print(f"\n---Actor {cur_player} sparta search---")
            move = actors[search_actor_idx].sparta_search(
                game, move, num_search, threshold
            )

        print(f"Active Player {cur_player} pick action: {move.to_string()}")
        moves.append(move)
        game.step(move)
        step += 1

    print(f"Final Score: {game.get_score()}, Seed: {seed}")
    return moves, game.get_score()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--save_dir", type=str, default="exps/sparta")
    parser.add_argument("--num_search", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--search_player", type=int, default=1)
    parser.add_argument("--seed", type=int, default=200191)
    parser.add_argument("--game_seed", type=int, default=19)
    parser.add_argument("--learned_belief", type=int, default=0)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--sad_legacy", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_search(args)

