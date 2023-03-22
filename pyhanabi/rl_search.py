# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
from typing import Tuple, Dict
import random
import time
import pprint
pprint = pprint.pprint
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import random

# c++ backend
import set_path
set_path.append_sys_path()
import rela
import hanalearn

import utils
import common_utils
import belief_model

CARD_ID_TO_STRING = np.array([
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "Y1",
    "Y2",
    "Y3",
    "Y4",
    "Y5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
])

ACTION_ID_TO_STRING = np.array([
    "Discard 0",
    "Discard 1",
    "Discard 2",
    "Discard 3",
    "Discard 4",
    "Play 0",
    "Play 1",
    "Play 2",
    "Play 3",
    "Play 4",
    "Reveal color R",
    "Reveal color Y",
    "Reveal color G",
    "Reveal color W",
    "Reveal color B",
    "Reveal rank 1",
    "Reveal rank 2",
    "Reveal rank 3",
    "Reveal rank 4",
    "Reveal rank 5",
    "INVALID"
])

ACTION_ID_TO_STRING_SHORT = np.array([
    "discard_0",
    "discard_1",
    "discard_2",
    "discard_3",
    "discard_4",
    "play_0",
    "play_1",
    "play_2",
    "play_3",
    "play_4",
    "hint_R",
    "hint_Y",
    "hint_G",
    "hint_W",
    "hint_B",
    "hint_1",
    "hint_2",
    "hint_3",
    "hint_4",
    "hint_5",
    "INVALID"
])


class SearchWrapper:
    def __init__(
        self,
        player_idx,
        public_belief,
        weight_file,
        partner_weight_file,
        belief_file,
        num_samples,
        explore_eps,
        n_step,
        gamma,
        train_device,
        rl_rollout_device,
        bp_rollout_device,
        belief_device,
        rollout_bsize,
        num_thread,
        num_game_per_thread,
        log_bsize_freq=-1,
        legacy_sad=False,
        legacy_sad_partner=False,
        replay_buffer=None,
        test_partner=1,
    ):
        self.player_idx = player_idx
        self.public_belief = public_belief
        assert not public_belief
        self.num_thread = num_thread
        self.num_game_per_thread = num_game_per_thread
        self.num_samples = num_samples
        self.acceptance_rate = 0.05
        self.legacy_sad = legacy_sad
        self.legacy_sad_partner = legacy_sad_partner
        self.replay_buffer = replay_buffer
        self.test_partner = test_partner

        if rl_rollout_device is None:
            self.rl = None
            self.rl_runner = None
        else:

            if legacy_sad:
                self.rl = utils.load_sad_model(
                        weight_file, 
                        train_device,
                        multi_step=1)
                config = {
                    "sad": True,
                    "hide_action": False,
                    "weight": weight_file,
                    "parameterized": False,
                    "sad_legacy": True,
                    "multi_step": 1,
                    "boltzmann_act": False,
                    "method": "iql",
                }
            else:
                # NOTE: multi-step is hard-coded to 1
                self.rl, config = utils.load_agent(
                    weight_file, {
                        "device": train_device, 
                        "off_belief": False,
                        "multi_step": 1
                    }
                )

            assert not config["hide_action"]
            assert not config["boltzmann_act"]
            assert config["method"] == "iql"
            assert self.rl.multi_step == 1

            self.rl_runner = rela.BatchRunner(
                self.rl.clone(rl_rollout_device),
                rl_rollout_device,
                rollout_bsize,
                ["act", "compute_priority"],
            )
            self.rl_runner.start()

        self.bp, config = self.create_bp_model(
                weight_file, legacy_sad, bp_rollout_device, 1)
        assert not config["hide_action"]
        assert not config["boltzmann_act"]
        assert config["method"] == "iql"
        assert self.bp.multi_step == 1

        self.bp_runner = rela.BatchRunner(
            self.bp,
            bp_rollout_device,
            rollout_bsize,
            ["act", "compute_target"],
        )

        if log_bsize_freq > 0:
            self.bp_runner.set_log_freq(log_bsize_freq)
        self.bp_runner.start()

        self.bp_partner = None
        self.bp_partner_runner = None

        if test_partner:
            self.bp_partner, config_partner = self.create_bp_model(
                    partner_weight_file, legacy_sad_partner, bp_rollout_device, 1)

            assert not config_partner["hide_action"]
            assert not config_partner["boltzmann_act"]
            assert config_partner["method"] == "iql"
            assert self.bp_partner.multi_step == 1

            self.bp_partner_runner = rela.BatchRunner(
                self.bp_partner,
                bp_rollout_device,
                rollout_bsize,
                ["act", "compute_target"],
            )

            if log_bsize_freq > 0:
                self.bp_partner_runner.set_log_freq(log_bsize_freq)

            self.bp_partner_runner.start()

        if belief_file:
            self.belief_model = belief_model.ARBeliefModel.load(
                belief_file,
                belief_device,
                hand_size=5,
                num_sample=num_samples,
                fc_only=False,
                mode="priv",
            )
            self.blueprint_belief = belief_model.ARBeliefModel.load(
                belief_file,
                belief_device,
                hand_size=5,
                num_sample=num_samples,
                fc_only=False,
                mode="priv",
            )
            self.belief_runner = rela.BatchRunner(
                self.belief_model, belief_device, rollout_bsize, ["observe", "sample"]
            )
            self.belief_runner.start()
        else:
            self.belief_runner = None

        self.explore_eps = explore_eps
        self.gamma = gamma
        self.n_step = n_step
        self.actor = None
        self.reset()

    def create_bp_model(self, weight_file, sad, device, multi_step):
        if sad:
            bp = utils.load_sad_model(
                    weight_file, 
                    device,
                    multi_step=multi_step)
            config = {
                "sad": True,
                "hide_action": False,
                "weight": weight_file,
                "parameterized": False,
                "sad_legacy": True,
                "multi_step": 1,
                "boltzmann_act": False,
                "method": "iql",
            }
        else:
            bp, config = utils.load_agent(
                weight_file, {
                    "device": device, 
                    "off_belief": False,
                    "multi_step": 1
                }
            )

        return bp, config

    def reset(self):
        self.actor = hanalearn.RLSearchActor(
            self.player_idx,
            self.bp_runner,
            self.bp_partner_runner,
            self.rl_runner,
            self.belief_runner,  # belief runner
            self.num_samples,  # num samples
            self.public_belief,  # public belief
            False,  # joint search
            self.explore_eps,
            self.n_step,
            self.gamma,
            random.randint(1, 999999),
            self.legacy_sad,
            self.legacy_sad_partner,
            self.replay_buffer,
            self.test_partner,
        )
        self.actor.set_compute_config(self.num_thread, self.num_game_per_thread)

    def update_rl_model(self, model):
        self.rl_runner.update_model(model)

    def reset_rl_to_bp(self):
        self.rl.online_net.load_state_dict(self.bp.online_net.state_dict())
        self.rl.target_net.load_state_dict(self.bp.online_net.state_dict())
        self.update_rl_model(self.rl)
        self.actor.reset_rl_rnn()


def main(args):
    pprint(vars(args))

    log_save_file = f"{args.player_name[1]}_vs_{args.player_name[0]}_game_{args.seed}.log"

    common_utils.set_all_seeds(args.seed)
    logger_path = os.path.join(args.save_dir, log_save_file)
    sys.stdout = common_utils.Logger(logger_path)
    print("log path:", logger_path)

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    print("explore eps:", explore_eps)
    print("mean eps:", np.mean(explore_eps))

    replay_buffer = None
    if args.save_game:
        replay_buffer = rela.RNNPrioritizedReplay(
            2,
            args.seed,
            1.0,  # priority exponent
            0.0,  # priority weight
            3, #prefetch
        )

    def create_search_wrapper(
            player_idx, 
            weight_file, 
            partner_weight_file, 
            rl_rollout_device,
            sad_legacy,
            sad_legacy_partner,
            test_partner):
        return SearchWrapper(
            player_idx,
            args.public_belief,
            weight_file,
            partner_weight_file,
            args.belief_file,
            args.num_samples,
            explore_eps,
            args.n_step,
            args.gamma,
            args.train_device,
            rl_rollout_device,
            args.bp_rollout_device,
            args.belief_device,
            args.rollout_batchsize,
            args.num_thread,
            args.num_game_per_thread,
            legacy_sad=sad_legacy,
            legacy_sad_partner=sad_legacy_partner,
            replay_buffer=replay_buffer,
            test_partner=test_partner,
        )

    search_wrappers = []
    rl_rollout_devices = [None, args.rl_rollout_device]
    test_partner = [1, 0]
    weight_files = [args.weight2, args.weight2]
    partner_weight = [args.weight1, None]
    sad_legacy = [args.sad_legacy[1], args.sad_legacy[1]]
    sad_legacy_partner = [args.sad_legacy[0], 0]
    for i in range(2):
        search_wrappers.append(create_search_wrapper(
            i, 
            weight_files[i], 
            partner_weight[i], 
            rl_rollout_devices[i], 
            sad_legacy[i],
            sad_legacy_partner[i],
            test_partner[i])
        )
        print()

    for i in range(2):
        partner_i = (i + 1) % 2
        search_wrappers[i].actor.set_partner(search_wrappers[partner_i].actor)

    actors = [wrapper.actor for wrapper in search_wrappers]

    now = datetime.now()

    game_data = run(
        args.game_seed,
        actors,
        search_wrappers[1],
        args,
    )

    if not args.save_game:
        return

    data = replay_to_dataframe(args, replay_buffer, now, game_data)

    if args.upload_gcloud:
        upload_gcloud(args, data, now)


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
        print("================STEP %d================\n" % step)
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
                    print("set use rl")
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

    for i, actor in enumerate(actors):
        print(f"\n---Actor {i} push episode to replay buffer---")
        actor.push_episode_to_replay_buffer()

    print(f"Final Score: {game.get_score()}, Seed: {seed}")

    return game_data


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
    bp_scores = search_actor.actor.run_sim_games(
        game, 
        args.num_eval_game, 
        0, 
        eval_seed, 
        sim_hands, 
        use_sim_hands
    )
    assert np.mean(bp_scores) <= max_possible_score + 1e-5
    if max_possible_score - np.mean(bp_scores) < args.threshold:
        return np.mean(bp_scores), 0

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


def replay_to_dataframe(args, replay_buffer, now, game_data):
    if replay_buffer is None:
        return
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    batch1, batch2 = replay_buffer.sample_from_list_split(2, "cpu", [0, 1])
    data = batch_to_dataset(args, batch1, batch2, date_time, game_data)

    return data

# ===============================================================

def batch_to_dataset(args, batch1, batch2, date_time, game_data):
    df = pd.DataFrame()

    print("player 0")
    obs_df = player_dataframe(args, batch1, 0, date_time, game_data[0])
    df = pd.concat([df, obs_df])

    print("player 1")
    obs_df = player_dataframe(args, batch2, 1, date_time, game_data[1])
    df = pd.concat([df, obs_df])

    df = df.reset_index(drop=True)

    if args.verbose:
        print("num cows:", df.shape[0])
        print("num columns:", len(list(df.columns.values)))

    columns = [
        # "game",
        "player",
        # "partner",
        "turn",
        "action",
        "rl_action",
        "rl_action_chosen",
        "rl_score",
        "bp_score",
        "rl_bp_diff",
        "rl_bp_diff",
        "diff_threshold",
        # "rl_actor",
        # "card_0",
        # "card_1",
        # "card_2",
        # "card_3",
        # "card_4",
        # "red_fireworks",
        # "yellow_fireworks",
        # "green_fireworks",
        # "white_fireworks",
        # "blue_fireworks",
    ]

    # pprint(df.columns.tolist())
    # print(df[columns].to_string(index=False))

    return df

def player_dataframe(args, batch, player, date_time, game_data):
    df = pd.DataFrame()

    # Add meta data
    meta_df = meta_data(args, batch, player, date_time)
    df = pd.concat([df, meta_df])

    # Add turn numbers
    hand_df = turn_data(args, batch)
    df = pd.concat([df, hand_df], axis=1)

    # Add observation
    obs_df = extract_obs(args, batch.obs, player)
    df = pd.concat([df, obs_df], axis=1)

    # Add legal moves
    legal_moves_df = extract_legal_moves(args, batch.obs["legal_move"])
    df = pd.concat([df, legal_moves_df], axis=1)

    # Add Action
    action_df = extract_column(args, batch.action["a"], "action")
    df = pd.concat([df, action_df], axis=1)

    # Add Q Values
    action_df = extract_q_values(args, batch.action["all_q"])
    df = pd.concat([df, action_df], axis=1)

    # Add Terminal
    terminal_df = extract_terminal(args, batch.terminal)
    df = pd.concat([df, terminal_df], axis=1)

    # Add bombs triggered
    df = add_bombs_triggered(args, df)

    # RL Search data
    if not args.skip_search:
        # Add RL Action
        action_df = extract_column(args, batch.action["rl_a"], "rl_action")
        df = pd.concat([df, action_df], axis=1)

        # Add RL Q Values
        action_df = extract_q_values(args, batch.action["rl_all_q"], "rl")
        df = pd.concat([df, action_df], axis=1)

        # Add RL Actor
        action_df = extract_column(args, batch.action["rl_actor"], "rl_actor")
        df = pd.concat([df, action_df], axis=1)

        rl_search_df = rl_search_data_to_df(args, game_data)
        df = pd.concat([df, rl_search_df], axis=1)

    # Remove rows after game has ended
    df = remove_states_after_terminal(args, df, batch.terminal)

    return df


def meta_data(args, batch, player, date_time):
    priv_s = batch.obs["priv_s"]
    num_rows = priv_s.shape[0] * priv_s.shape[1]

    game_names = []

    for i in range(priv_s.shape[1]):
        game_names.append(f"{args.player_name[0]}_vs_{args.player_name[1]}_game_{i}")

    data = np.array(game_names, )
    data = np.repeat(data, priv_s.shape[0])
    data = np.reshape(data, (num_rows, 1))

    meta_data = np.array([
        args.player_name[player],
        args.player_name[(player + 1) % 2],
        args.data_type,
        date_time
    ], dtype=str)

    meta_data = np.tile(meta_data, (num_rows, 1))
    data = np.concatenate((data, meta_data), axis=1)

    labels = [
        "game",
        "player",
        "partner",
        "data_type",
        "datetime",
    ]

    return pd.DataFrame(
        data=data,
        columns=labels
    )


def turn_data(args, batch):
    shape = batch.obs["priv_s"].shape
    data = np.arange(0,80, dtype=np.uint8)
    data = np.tile(data, (shape[1], 1))
    data = np.reshape(data, (shape[0] * shape[1],))
    labels = ["turn"]

    return pd.DataFrame(
        data=data,
        columns=labels
    )


def extract_obs(args, obs, player):
    df = pd.DataFrame()

    if args.sad_legacy[player]:
        # Make sad priv_s the same as OBL priv_s
        priv_s = obs["priv_s"][:, :, 125:783]
    else:
        priv_s = obs["priv_s"]

    partner_hand_idx = 125
    missing_cards_idx = 127
    board_idx = 203
    discard_idx = 253
    last_action_idx = 308
    v0_belief_idx = 658

    # Own hand
    hand_df = extract_hand(args, obs["own_hand_ar"], "")
    df = pd.concat([df, hand_df], axis=1)

    # Partner Hand
    partner_hand = np.array(priv_s[:, :, :partner_hand_idx])
    hand_df = extract_hand(args, partner_hand, "partner_")
    df = pd.concat([df, hand_df], axis=1)

    # Hands missing Card
    missing_cards = np.array(priv_s[:, :, partner_hand_idx:missing_cards_idx])
    missing_cards_df = extract_missing_cards(args, missing_cards)
    df = pd.concat([df, missing_cards_df], axis=1)

    # Board
    board = np.array(priv_s[:, :, missing_cards_idx:board_idx])
    board_df = extract_board(args, board)
    df = pd.concat([df, board_df], axis=1)

    # Discards
    discards = np.array(priv_s[:, :, board_idx:discard_idx])
    discards_df = extract_discards(args, discards)
    df = pd.concat([df, discards_df], axis=1)

    # Last Action
    last_action = np.array(priv_s[:, :, discard_idx:last_action_idx])
    last_action_df = extract_last_action(args, last_action)
    df = pd.concat([df, last_action_df], axis=1)

    # Knowledge
    card_knowledge = np.array(priv_s[:, :, last_action_idx:v0_belief_idx])
    card_knowledge_df = extract_card_knowledge(args, card_knowledge)
    df = pd.concat([df, card_knowledge_df], axis=1)

    return df


def extract_hand(args, hand, label_str):
    hand = np.array(hand, dtype=int)
    shape = hand.shape
    hand = np.reshape(hand, (shape[0], shape[1], 5, 25))
    hand = np.swapaxes(hand, 0, 1) 
    cards = np.argmax(hand, axis=3)
    cards = np.reshape(cards, (cards.shape[0] * cards.shape[1], 5))
    cards = cards.astype(np.uint8)

    labels = []
    for i in range(5):
        labels.append(f"{label_str}card_{i}")

    # cards = CARD_ID_TO_STRING[cards]

    return pd.DataFrame(
        data=cards,
        columns=labels
    )


def extract_missing_cards(args, missing_cards):
    missing_cards = np.array(missing_cards, dtype=np.uint8)
    missing_cards = np.swapaxes(missing_cards, 0, 1)
    num_rows = missing_cards.shape[0] * missing_cards.shape[1]
    missing_cards = np.reshape(missing_cards, (num_rows, missing_cards.shape[2]))

    labels = ["own_missing_card", "partner_missing_card"]

    return pd.DataFrame(
        data=missing_cards,
        columns=labels
    )

def extract_board(args, board):
    num_rows = board.shape[0] * board.shape[1]
    board = np.array(board, dtype=np.uint8)
    board = np.swapaxes(board, 0, 1)

    # Encoding positions
    deck_idx = 40
    fireworks_idx = 65
    info_idx = 73
    life_idx = 76

    board_data = np.empty((num_rows, 0), dtype=np.uint8)

    # Deck
    deck = board[:, :, :deck_idx]
    deck_size = deck.sum(axis=2)
    deck_size = np.expand_dims(deck_size, axis=2)
    deck_size = np.reshape(deck_size, (num_rows, deck_size.shape[2]))
    board_data = np.concatenate((board_data, deck_size), axis=1)

    # Fireworks
    fireworks = board[:, :, deck_idx:fireworks_idx]
    fireworks = np.reshape(fireworks, (fireworks.shape[0], fireworks.shape[1], 5, 5))
    non_empty_piles = np.sum(fireworks, axis=3)
    empty_piles = non_empty_piles ^ (non_empty_piles & 1 == non_empty_piles)
    fireworks = np.argmax(fireworks, axis=3) + 1 - empty_piles
    fireworks = np.reshape(fireworks, (num_rows, fireworks.shape[2]))
    fireworks = fireworks.astype(np.uint8)
    board_data = np.concatenate((board_data, fireworks), axis=1)

    # Info Tokens
    info = board[:, :, fireworks_idx:info_idx]
    info_tokens = info.sum(axis=2)
    info_tokens = np.expand_dims(info_tokens, axis=2)
    info_tokens = np.reshape(info_tokens, (num_rows, info_tokens.shape[2]))
    board_data = np.concatenate((board_data, info_tokens), axis=1)

    # Life Tokens
    lives = board[:, :, info_idx:life_idx]
    lives = lives.sum(axis=2)
    lives = np.expand_dims(lives, axis=2)
    lives = np.reshape(lives, (num_rows, lives.shape[2]))
    board_data = np.concatenate((board_data, lives), axis=1)

    # Column labels
    labels = ["deck_size"]
    for colour in ["red", "yellow", "green", "white", "blue"]:
        labels.append(f"{colour}_fireworks")
    labels.extend(["info_tokens", "lives"])

    return pd.DataFrame(
        data=board_data,
        columns=labels
    )


def extract_discards(args, discards):
    num_rows = discards.shape[0] * discards.shape[1]
    discards = np.array(discards, dtype=np.uint8)
    discards = np.swapaxes(discards, 0, 1)
    discards_data = np.empty((num_rows, 0), dtype=np.uint8)

    idx_pos_per_rank = [3, 5, 7, 9, 10]
    num_cards_per_rank = [3, 2, 2, 2, 1]
    colours = ["red", "yellow", "green", "white", "blue"]

    bits_per_colour = 10

    labels = []

    for i, colour in enumerate(["red", "yellow", "green", "white", "blue"]):
        offset = i * bits_per_colour

        for j in range(5):
            labels.append(f"{colour}_{j + 1}_discarded")

            end_pos = offset + idx_pos_per_rank[j]
            start_pos = end_pos - num_cards_per_rank[j]
            num_discards = discards[:, :, start_pos:end_pos]
            num_discards = np.sum(num_discards, axis=2)
            num_discards = np.expand_dims(num_discards, axis=2)
            num_discards = np.reshape(num_discards, (num_rows, num_discards.shape[2]))
            discards_data = np.concatenate((discards_data, num_discards), axis=1)

    return pd.DataFrame(
        data=discards_data,
        columns=labels
    )

def extract_last_action(args, last_action):
    num_rows = last_action.shape[0] * last_action.shape[1]
    last_action = np.array(last_action, dtype=np.uint8)
    last_action = np.swapaxes(last_action, 0, 1)

    acting_player_idx = 2
    move_type_idx = 6
    target_player_idx = 8
    colour_revealed_idx = 13
    rank_revealed_idx = 18
    reveal_outcome_idx = 23
    card_position_idx = 28
    card_played_idx = 53
    card_played_scored_idx = 54

    move_type = last_action[:, :, acting_player_idx:move_type_idx]
    card_position = last_action[:, :, reveal_outcome_idx:card_position_idx]
    colour_revealed = last_action[:, :, target_player_idx:colour_revealed_idx]
    rank_revealed = last_action[:, :, colour_revealed_idx:rank_revealed_idx]
    card_played_scored = last_action[:, :, card_played_idx:card_played_scored_idx]

    action_index = [1,0,2,3]
    move_index = range(5)
    action_functions = [card_position, card_position, colour_revealed, rank_revealed]

    conditions = []
    for action_i in action_index:
        for move_i in move_index:
            conditions.append((move_type[:, :, action_i] == 1) & \
                              (action_functions[action_i][:, :, move_i] == 1))
    conditions.append(True)

    move_id = range(21)
    last_action_data = np.select(conditions, move_id, default=20)
    last_action_data = np.expand_dims(last_action_data, axis=2)


    last_action_data = np.concatenate((last_action_data, card_played_scored), axis=2)
    last_action_data = np.reshape(last_action_data, (num_rows, last_action_data.shape[2]))

    return pd.DataFrame(
        data=last_action_data,
        columns=["last_action", "last_action_scored"]
    )


def extract_card_knowledge(args, card_knowledge):
    num_rows = card_knowledge.shape[0] * card_knowledge.shape[1]
    card_knowledge = np.array(card_knowledge)
    card_knowledge = np.swapaxes(card_knowledge, 0, 1)
    card_knowledge = np.reshape(card_knowledge, (num_rows, card_knowledge.shape[2]))

    possible_cards_len = 25
    colour_hinted_len = 5
    rank_hinted_len = 5
    card_len = possible_cards_len + colour_hinted_len + rank_hinted_len
    player_len = card_len * 5

    labels = []

    players = ["", "partner_"]
    colours = "RYGWB"

    for player in range(2):
        for card in range(5):
            for colour in range(5):
                for rank in range(5):
                    labels.append(f"{players[player]}card_{card}_{colours[colour]}{rank+1}_belief")

            for colour in range(5):
                labels.append(f"{players[player]}card_{card}_{colours[colour]}_hinted")

            for rank in range(5):
                labels.append(f"{players[player]}card_{card}_{rank + 1}_hinted")


    return pd.DataFrame(
        data=card_knowledge,
        columns=labels
    )


def extract_legal_moves(args, legal_move):
    num_rows = legal_move.shape[0] * legal_move.shape[1]
    legal_move = np.array(legal_move, dtype=np.uint8)
    legal_move = np.swapaxes(legal_move, 0, 1)
    legal_move = np.reshape(legal_move, (num_rows, legal_move.shape[2]))

    labels=[]

    for move_id in range(21):
        labels.append(f"legal_move_{ACTION_ID_TO_STRING_SHORT[move_id]}")

    df = pd.DataFrame(
        data=legal_move,
        columns=labels
    )

    return df

def extract_column(args, action, name):
    num_rows = action.shape[0] * action.shape[1]
    action = np.array(action, dtype=np.uint8)
    action = np.swapaxes(action, 0, 1)
    action = np.expand_dims(action, axis=2)
    action = np.reshape(action, (num_rows, action.shape[2]))

    return pd.DataFrame(
        data=action,
        columns=[name]
    )


def extract_q_values(args, q_values, prefix=""):
    num_rows = q_values.shape[0] * q_values.shape[1]
    q_values = np.array(q_values)
    q_values = np.swapaxes(q_values, 0, 1)
    q_values = np.reshape(q_values, (num_rows, q_values.shape[2]))
    prefix = prefix if len(prefix) == 0 else f"{prefix}_"

    labels = []
    for move_id in range(21):
        labels.append(f"{prefix}q_value_move_{ACTION_ID_TO_STRING_SHORT[move_id]}")

    return pd.DataFrame(
        data=q_values,
        columns=labels
    )


def extract_terminal(args, terminal):
    num_rows = terminal.shape[0] * terminal.shape[1]
    terminal = np.array(terminal, dtype=np.uint8)
    terminal = np.swapaxes(terminal, 0, 1)
    terminal = np.expand_dims(terminal, axis=2)
    terminal = np.reshape(terminal, (num_rows, terminal.shape[2]))

    return pd.DataFrame(
        data=terminal,
        columns=["terminal"]
    )


def add_bombs_triggered(args, df):
    action = df["action"]
    cards = np.array([ df[f"card_{i}"] for i in range(5) ])
    card_to_colour = np.repeat(np.arange(0,5),5)
    colours = ["red", "yellow", "green", "white", "blue"]
    colour_to_fireworks = np.array([ df[f"{colours[i]}_fireworks"] for i in range(5) ])
    card_to_rank = np.array(list(np.arange(1,6)) * 5)

    condition = []
    for card_position in range(5):
        for colour in range(5):
            condition.append(
                (action == 5 + card_position)
                & (card_to_colour[cards[card_position]] == colour) 
                & (colour_to_fireworks[colour] + 1 != card_to_rank[cards[card_position]]),
            )

    result = [1] * len(condition)

    last_action_data = np.select(condition, result, default=0)

    bombs_triggered_df = pd.DataFrame(
        data=last_action_data,
        columns=["action_trigger_bomb"],
    )
    df = pd.concat([df, bombs_triggered_df], axis=1)

    df["last_action_trigger_bomb"] = np.where(
        (df["last_action"] >= 5)
        & (df["last_action"] <= 9)
        & (df["last_action_scored"] == 0), 1, 0
    )

    return df

def rl_search_data_to_df(args, game_data):
    df = pd.DataFrame()

    for key, value in game_data.items():
        df[key] = value

    return df

def remove_states_after_terminal(args, df, terminal):
    terminal =  np.array(terminal, dtype=np.uint8)
    terminal = np.swapaxes(terminal, 0, 1)
    terminal = np.expand_dims(terminal, axis=2)
    inv_terminal = terminal ^ (terminal & 1 == terminal)
    sum = np.sum(inv_terminal, axis=1)
    rows = np.array(range(sum.shape[0]))
    rows = np.expand_dims(rows, axis=1)
    sumrows = np.hstack((rows, sum))
    sumrows = sumrows.astype(int)
    sumrows = sumrows[sumrows[:,1] < terminal.shape[1]]
    terminal[sumrows[:,0], sumrows[:,1], 0] = 0
    num_rows = terminal.shape[0] * terminal.shape[1]
    remove_rows = np.reshape(terminal, (num_rows, terminal.shape[2]))
    remove_rows = remove_rows.astype(bool)

    remove_rows_df = pd.DataFrame(
        data=remove_rows,
        columns=["remove_rows"],
    )
    df = pd.concat([df, remove_rows_df], axis=1)
    df = df[~df.remove_rows]
    df = df.drop("remove_rows", axis=1)
    return df


def upload_gcloud(args, data, now):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    date_time = now.strftime("%m.%d.%Y_%H:%M:%S")

    filename = f"{args.player_name[1]}_vs_{args.player_name[0]}_{date_time}.pkl"
    filepath = os.path.join(args.save_dir, filename)

    print("Saving:", filepath)
    data.to_pickle(filepath, compression="gzip")



# ===============================================================



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
    parser.add_argument("--weight1", type=str, required=True)
    parser.add_argument("--weight2", type=str, required=True)
    parser.add_argument("--player_name", type=str, required=True)
    parser.add_argument("--sad_legacy", type=str, default="0,0")

    parser.add_argument("--belief_file", type=str, default="")
    parser.add_argument("--belief_device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--maintain_exact_belief", type=int, default=1)
    parser.add_argument("--search_exact_belief", type=int, default=1)

    parser.add_argument("--skip_search", type=int, default=0)
    parser.add_argument("--upload_gcloud", type=int, default=0)
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--ad_hoc", type=int, default=0)
    parser.add_argument("--save_game", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="game_data/default")

    args = parser.parse_args()
    if args.debug:
        args.num_epoch = 1
        args.epoch_len = 200
        args.num_eval_game = 500

    # Convert sad_legacy to valid list of ints
    args.sad_legacy = [int(x) for x in args.sad_legacy.split(",")]
    assert(len(args.sad_legacy) <= 2)
    if (len(args.sad_legacy) == 1):
        args.sad_legacy *= 2

    args.player_name = [x for x in args.player_name.split(",")]

    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    main(args)

