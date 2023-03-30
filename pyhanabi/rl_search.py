# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import random
import time
import pprint
pprint = pprint.pprint
import json
import torch
import numpy as np
from datetime import datetime
import pathlib

# c++ backend
import set_path
set_path.append_sys_path()
import rela
import hanalearn

import utils
import common_utils

from search_wrapper import SearchWrapper
from replay_buffer_to_dataframe import replay_to_dataframe
from google_cloud_handler import GoogleCloudHandler


def run_rl_search(args):
    pprint(vars(args))
    log_save_file = f"{args.player_name[1]}_vs_{args.player_name[0]}_game_{args.seed}.log"

    common_utils.set_all_seeds(args.seed)
    logger_path = os.path.join(args.save_dir, "logs", log_save_file)
    sys.stdout = common_utils.Logger(logger_path)
    print("log path:", logger_path)

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    # print("explore eps:", explore_eps)
    # print("mean eps:", np.mean(explore_eps))

    replay_buffer = None
    if args.save_game:
        replay_buffer = rela.RNNPrioritizedReplay(
            2,
            args.seed,
            1.0,  # priority exponent
            0.0,  # priority weight
            3, #prefetch
        )

    # Blueprint model
    bp_weight_files = [args.search_partner_weight, [args.weight]]
    bp_sad_legacy = [args.search_partner_sad_legacy, [args.sad_legacy]]

    # Test partner model
    test_partner_weight_file = [args.test_partner_weight, None]
    test_partner_sad_legacy = [args.test_partner_sad_legacy, 0]
    is_test_partner = [1, 0]

    # RL models
    rl_weight_file = [None, args.weight]
    rl_sad_legacy = [0, args.sad_legacy]
    rl_rollout_device = [None, args.rl_rollout_device]

    search_wrapper = []
    for i in range(2):
        print("Setup Wrapper", i)
        search_wrapper.append(SearchWrapper(
            i,
            args.public_belief,
            bp_weight_files[i],
            bp_sad_legacy[i],
            test_partner_weight_file[i],
            test_partner_sad_legacy[i],
            rl_weight_file[i],
            rl_sad_legacy[i],
            args.belief_file,
            args.num_samples,
            explore_eps,
            args.n_step,
            args.gamma,
            args.train_device,
            rl_rollout_device[i],
            args.bp_rollout_device,
            args.belief_device,
            args.rollout_batchsize,
            args.num_thread,
            args.num_game_per_thread,
            replay_buffer=replay_buffer,
            is_test_partner=is_test_partner[i],
        ))
        print()

    search_wrapper[0].actor.set_partner(search_wrapper[1].actor)
    search_wrapper[1].actor.set_partner(search_wrapper[0].actor)

    actors = [wrapper.actor for wrapper in search_wrapper]

    now = datetime.now()

    game_data, score = run(
        args.game_seed,
        actors,
        search_wrapper[1],
        args,
    )

    if not args.save_game:
        return

    data = replay_to_dataframe(args, replay_buffer, now, game_data)

    save_and_upload(args, data, score, now)


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def run(seed, actors, search_actor, args):
    game_data = [{ "rl_action_chosen": [],
                   "rl_score": [],
                   "bp_score": [],
                   "rl_bp_diff": [],
                   "diff_threshold": []} for x in range(2) ]

    params = {
        "players": str(len(actors)),
        "seed": str(seed),
        "bomb": str(0),
        "hand_size": str(5),
        "max_information_tokens": str(args.num_hint),  # global variable
        "random_start_player": str(0),  # do not randomize start_player
    }
    game = hanalearn.GameSimulator(params)
    step = 0

    # created once, reused for the rest of training
    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        random.randint(1, 999999),
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    for i, actor in enumerate(actors):
        print(f"---Actor {i} initialise---")
        actor.initialise()

    while not game.terminal():
        print("\n================STEP %d================\n" % step)
        if args.verbose:
            print(f"{game.state().to_string()}\n")

        if not args.skip_search:
            using_rl = search_actor.actor.using_rl()
            assert using_rl >= 0
        cur_player = game.state().cur_player()

        if not args.skip_search:
            for i in range(2):
                game_data[i]["rl_action_chosen"].append(0)
                game_data[i]["rl_score"].append(0)
                game_data[i]["bp_score"].append(0)
                game_data[i]["rl_bp_diff"].append(0)
                game_data[i]["diff_threshold"].append(0)
            for i, actor in enumerate(actors):
                if i != search_actor.player_idx:
                    continue
                if args.maintain_exact_belief:
                    print(f"\n---Actor {i} update belief---")
                    actor.update_belief(game, args.ad_hoc)

            # if already in rl mode, then no more training
            if (
                args.num_rl_step > 0
                and using_rl == 0
                and cur_player == search_actor.player_idx
            ):
                replay_buffer.clear()
                # always restart from bp
                search_actor.reset_rl_to_bp()
                t = time.time()
                bp_score, rl_score = train(
                    game, search_actor, replay_buffer, 
                    args, random.randint(1, 999999)
                )

                print(
                    "rl - bp:  %.4f, threshold: %.4f, time taken: %ds" % (
                        rl_score - bp_score, args.threshold, time.time() - t))
                    
                if rl_score - bp_score >= args.threshold:
                    game_data[i]["rl_action_chosen"][-1] = 1
                    print("Using rl move")
                    search_actor.actor.set_use_rl(args.num_rl_step)

                game_data[i]["rl_score"][-1] = rl_score
                game_data[i]["bp_score"][-1] = bp_score
                game_data[i]["rl_bp_diff"][-1] = rl_score - bp_score
                game_data[i]["diff_threshold"][-1] = args.threshold

        for i, actor in enumerate(actors):
            print(f"\n---Actor {i} observe---")
            actor.observe(game)

        if not args.skip_search \
           and search_actor.belief_runner is not None:
                for i, actor in enumerate(actors):
                    if i != search_actor.player_idx:
                        continue
                    print(f"\n---Actor {i} update belief hid---")
                    actor.update_belief_hid(game)

        actions = []
        for i, actor in enumerate(actors):
            print(f"\n---Actor {i} decide action---")
            action = actor.decide_action(game)
            actions.append(action)
            for label, move in actor.get_chosen_moves().items():
                print(f"{label}:", move)

        move = game.get_move(actions[cur_player])

        if not args.skip_search \
           and args.sparta \
           and cur_player == search_actor.player_idx:
            move = actor.sparta_search(
                game, move, args.sparta_num_search, args.sparta_threshold
            )

        print(f"\nActive Player {cur_player} pick action: {move.to_string()}")
        game.step(move)
        step += 1

        for i, actor in enumerate(actors):
            print(f"\n---Actor {i} observe after act---")
            actor.observe_after_act(game)

        if i == 1:
            sys.exit

    for i, actor in enumerate(actors):
        print(f"\n---Actor {i} push episode to replay buffer---")
        actor.push_episode_to_replay_buffer()

    score = game.get_score()

    print(f"Final Score: {score}, Seed: {seed}")

    return game_data, score


def train(game, search_actor, replay_buffer, args, eval_seed):
    if args.search_exact_belief:
        sim_hands = [[[]]]
        use_sim_hands = False
    else:
        sim_hands = search_actor.actor.sample_hands(game.state(), args.num_samples)
        use_sim_hands = True
        if len(sim_hands[0]) < search_actor.num_samples * search_actor.acceptance_rate:
            print(
                f"Belief acceptance rate is less than {search_actor.acceptance_rate}; "
                f"falling back to blueprint"
            )
            return None, None

    max_possible_score = game.state().max_possible_score()
    print("START RUN SIM GAMES ####################################")
    bp_scores = search_actor.actor.run_sim_games(
        game, 
        args.num_eval_game, 
        0, 
        eval_seed, 
        sim_hands, 
        use_sim_hands
    )
    print("END RUN SIM GAMES ####################################")
    assert np.mean(bp_scores) <= max_possible_score + 1e-5
    # if max_possible_score - np.mean(bp_scores) < args.threshold:
        # return np.mean(bp_scores), 0

    print("STARTING DATA GENERATION ####################################")
    search_actor.actor.start_data_generation(
        game, replay_buffer, args.num_rl_step, sim_hands, use_sim_hands, False
    )

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("Done: replay buffer size:", replay_buffer.size())

    optim = torch.optim.Adam(
        search_actor.rl.online_net.parameters(), lr=args.lr, eps=args.eps
    )

    if args.final_only:
        for p in search_actor.rl.online_net.parameters():
            p.requires_grad = False
        for p in search_actor.rl.online_net.fc_v.parameters():
            p.requires_grad = True
        for p in search_actor.rl.online_net.fc_a.parameters():
            p.requires_grad = True
        for p in search_actor.rl.online_net.pred_1st.parameters():
            p.requires_grad = True

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    saver = common_utils.TopkSaver(args.save_dir, 5)

    for epoch in range(args.num_epoch):
        tachometer.start()
        stat.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                search_actor.rl.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                search_actor.update_rl_model(search_actor.rl)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")
            loss, priority, _ = search_actor.rl.loss(batch, args.aux, stat)
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                search_actor.rl.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            replay_buffer.update_priority(priority)
            stopwatch.time("other")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        print("EPOCH: %d" % epoch)
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, 1)
        stat.summary(epoch)
        stopwatch.summary()

    search_actor.actor.stop_data_generation()
    search_actor.update_rl_model(search_actor.rl)
    rl_scores = search_actor.actor.run_sim_games(
        game, 
        args.num_eval_game, 
        args.num_rl_step, 
        eval_seed, 
        sim_hands, 
        use_sim_hands
    )

    rl_mean = np.mean(rl_scores)
    rl_sem = np.std(rl_scores) / np.sqrt(len(rl_scores))
    bp_mean = np.mean(bp_scores)
    bp_sem = np.std(bp_scores) / np.sqrt(len(bp_scores))
    print(f">>>>>bp score: {bp_mean:.3f} +/- {bp_sem:.3f}")
    print(f">>>>>rl score: {rl_mean:.3f} +/- {rl_sem:.3f}")
    print(f"mean diff: {rl_mean - bp_mean}")
    print(f"mean-sem diff: {rl_mean - rl_sem - bp_mean}")
    print(f"mean-(sem+sem) diff: {rl_mean - rl_sem - bp_mean - bp_sem}")

    return bp_mean, rl_mean


def save_and_upload(args, data, score, now):
    if not args.save_game:
        return

    #Create folder
    games_dir = os.path.join(args.save_dir, "games")
    scores_dir = os.path.join(args.save_dir, "scores")
    logs_dir = os.path.join(args.save_dir, "logs")
    if not os.path.exists(games_dir):
        os.makedirs(games_dir)
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    filename = f"{args.player_name[1]}_vs_{args.player_name[0]}_game_{args.seed}"
    game_file = f"{filename}.pkl"
    score_file = f"{filename}.txt"
    log_file = f"{filename}.log"
    game_path = os.path.join(games_dir, game_file)
    score_path = os.path.join(scores_dir, score_file)
    log_path = os.path.join(logs_dir, log_file)

    print("saving:", game_path)
    data.to_pickle(game_path, compression="gzip")

    print("saving:", score_path)
    with open(score_path, 'w') as f:
      f.write("%d\n" % score)
    
    if not args.upload_gcloud:
        return

    hanabi_dir = "hanabi-search-games"
    game_path_obj = pathlib.Path(game_path)
    gc_game_path = os.path.join(args.gcloud_dir, *game_path_obj.parts[1:])
    score_path_obj = pathlib.Path(score_path)
    gc_score_path = os.path.join(args.gcloud_dir, *score_path_obj.parts[1:])
    log_path_obj = pathlib.Path(log_path)
    gc_log_path = os.path.join(args.gcloud_dir, *log_path_obj.parts[1:])

    gc_handler = GoogleCloudHandler("aiml-reid-research", "Ravi")

    print("uploading:", gc_game_path)
    gc_handler.assert_file_doesnt_exist(gc_game_path)
    gc_handler.upload(game_path, gc_game_path)

    print("uploading:", gc_score_path)
    gc_handler.assert_file_doesnt_exist(gc_score_path)
    gc_handler.upload(score_path, gc_score_path)

    print("uploading:", gc_log_path)
    gc_handler.assert_file_doesnt_exist(gc_log_path)
    gc_handler.upload(log_path, gc_log_path)


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--public_belief", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--game_seed", type=int, default=0)
    parser.add_argument("--n_step", type=int, default=1, help="n_step return")
    parser.add_argument("--num_eval_game", type=int, default=5000)
    parser.add_argument("--final_only", type=int, default=0)
    parser.add_argument("--sparta", type=int, default=0)
    parser.add_argument("--sparta_num_search", type=int, default=10000)
    parser.add_argument("--sparta_threshold", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_rl_step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")

    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--burn_in_frames", type=int, default=5000)
    parser.add_argument("--priority_exponent", type=float, default=0.9, help="alpha")
    parser.add_argument("--priority_weight", type=float, default=0.6, help="beta")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--rl_rollout_device", type=str, default="cuda:1")
    parser.add_argument("--bp_rollout_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--rollout_batchsize", type=int, default=8000)

    parser.add_argument("--num_thread", type=int, default=1, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    parser.add_argument("--aux", type=float, default=0.25)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--epoch_len", type=int, default=5000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    parser.add_argument("--num_hint", type=int, default=8)
    parser.add_argument("--player_name", type=str, required=True)

    parser.add_argument("--belief_file", type=str, default="")
    parser.add_argument("--belief_device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--maintain_exact_belief", type=int, default=1)
    parser.add_argument("--search_exact_belief", type=int, default=1)

    parser.add_argument("--skip_search", type=int, default=0)
    parser.add_argument("--upload_gcloud", type=int, default=0)
    parser.add_argument("--gcloud_dir", type=str, default="hanabi-search-games")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--ad_hoc", type=int, default=0)
    parser.add_argument("--save_game", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="game_data/default")
    parser.add_argument("--split_type", type=str, default="six")

    # Model weights
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--sad_legacy", type=int, required=True)
    parser.add_argument("--test_partner_weight", type=str, required=True)
    parser.add_argument("--test_partner_sad_legacy", type=int, required=True)
    parser.add_argument("--search_partner_weight", type=str, required=True)
    parser.add_argument("--search_partner_sad_legacy", type=str, required=True)

    args = parser.parse_args()

    if args.debug:
        args.num_epoch = 1
        args.epoch_len = 200
        args.num_eval_game = 500

    args.player_name = [x for x in args.player_name.split(",")]

    args.search_partner_weight = [ 
        x for x in args.search_partner_weight.split(",") ]
    args.search_partner_sad_legacy = [
        int(x) for x in args.search_partner_sad_legacy.split(",") ]

    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    run_rl_search(args)

