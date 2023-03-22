# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import json
import numpy as np
import torch
import sys
import random
import pprint
pprint = pprint.pprint

from create import *
import rela
import r2d2 
import utils
import gc

def evaluate(
    agents,
    num_game,
    seed,
    bomb,
    eps,
    sad,
    hide_action,
    *,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    convention=[],
    convention_indexes=None,
    override=[0, 0],
    act_parameterized=[0, 0],
    num_parameters=0,
    belief_stats=False,
    belief_model_path="None",
    partners=None,
    sad_legacy=[0, 0],
    shuffle_colour=[0, 0],
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    num_player = len(agents) 
    if not isinstance(hide_action, list):
        hide_action = [hide_action for _ in range(num_player)]
    if not isinstance(sad, list):
        sad = [sad for _ in range(num_player)]

    runners = []
    for i, agent in enumerate(agents):
        runners.append(rela.BatchRunner(agent.clone(device), device, 1000, ["act"]))

    partner_runners = []
    if partners is not None:
        for partner in partners:
            agent = partner["agent"]
            partner_runners.append(
                rela.BatchRunner(agent.clone(device), device, 1000, ["act"])
            )

    belief_runner = None
    if belief_stats:
        belief_runner = create_belief_runner(belief_model_path, device)

    context = rela.Context()
    threads = []

    games = create_envs(num_game, seed, num_player, bomb, max_len)

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []

    partner_idx = 0
    convention_index = 0

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []

            for i in range(num_player):
                runner = runners[i]
                sad_setting = sad[i]
                hide_action_setting = hide_action[i]
                act_parameterized_setting = act_parameterized[i]
                sad_legacy_setting = sad_legacy[i]

                if i > 0 and partners is not None:
                    runner = partner_runners[partner_idx]
                    sad_setting = partners[partner_idx]["sad"]
                    hide_action_setting = partners[partner_idx]["hide_action"]
                    act_parameterized_setting = partners[partner_idx]["parameterized"]
                    sad_legacy_setting = partners[partner_idx]["sad_legacy"]

                if convention_indexes is not None:
                    convention_index = convention_indexes[i]

                actor = hanalearn.R2D2Actor(
                    runner, # runner
                    num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    sad_setting, # sad
                    hide_action_setting, # hideAction
                    # convention, # convention
                    # act_parameterized_setting, # act parameterized
                    # convention_index, # conventionIndex
                    # override[i], # conventionOverride
                    # belief_stats, # beliefStats
                    sad_legacy_setting, # sadLegacy
                    # shuffle_colour[i], #shuffleColor
                )

                if belief_stats:
                    if belief_runner is None:
                        actor.set_belief_runner_stats(None)
                    else:
                        actor.set_belief_runner_stats(belief_runner)

                actors.append(actor)
                all_actors.append(actor)

            for i in range(num_player):
                act_partners = actors[:]
                act_partners[i] = None
                actors[i].set_partners(act_partners)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])
            if partners is not None:
                partner_idx = (partner_idx + 1) % len(partners)

            if convention_indexes is None and len(convention) > 0:
                convention_index = (convention_index + 1) % len(convention)
            elif convention_indexes is None and num_parameters > 0:
                convention_index = (convention_index + 1) % num_parameters

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        if runner is not None:
            runner.start()

    for runner in partner_runners:
        runner.start()

    if belief_runner is not None:
        belief_runner.start()

    context.start()
    context.join()

    for runner in runners:
        if runner is not None:
            runner.stop()

    for runner in partner_runners:
        runner.stop()

    if belief_runner is not None:
        belief_runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect, all_actors


def create_belief_runner(belief_model_path, device):
    belief_model = None

    if belief_model_path != "None":
        from belief_model import ARBeliefModel

        belief_config = utils.get_train_config(belief_model_path)

        belief_model = ARBeliefModel.load(
            belief_model_path,
            device,
            5,
            10,
            belief_config["fc_only"],
            belief_config["parameterized"],
            belief_config["num_conventions"],
        )

    belief_runner = None

    if belief_model is not None:
        belief_runner = rela.BatchRunner(
                belief_model, belief_model.device, 5000, ["sample"])


    return belief_runner


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
    device="cuda:0",
    convention="None",
    convention_indexes=None,
    override=[0, 0],
    belief_stats=False,
    belief_model="None",
    partner_models_path="None",
    pre_loaded_data=None,
    sad_legacy=[0, 0],
    partner_model_type="train",
):
    if pre_loaded_data is None:
        pre_loaded_data = load_agents(
            weight_files,
            overwrite=overwrite,
            device=device,
            belief_stats=belief_stats,
            partner_models_path=partner_models_path,
            sad_legacy=sad_legacy
        )

    agents = pre_loaded_data["agents"]
    sad = pre_loaded_data["sad"]
    hide_action = pre_loaded_data["hide_action"]
    parameterized = pre_loaded_data["parameterized"]
    belief_model_path = pre_loaded_data["belief_model_path"]

    if belief_model != "None" and belief_stats:
        belief_model_path = belief_model

    partners = load_partner_agents(partner_models_path, partner_model_type, True)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p, games = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,  # eps
            sad,
            hide_action,
            device=device,
            convention=load_json_list(convention),
            convention_indexes=convention_indexes,
            override=override,
            belief_stats=belief_stats,
            belief_model_path=belief_model_path,
            partners=partners,
            act_parameterized=parameterized,
            sad_legacy=sad_legacy,
        )
        scores.extend(score)
        perfect += p

    for agent in agents:
        agent.to("cpu")
        del agent
    torch.cuda.empty_cache()
    gc.collect()

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %.3f +/- %.3f" % (mean, sem),
            "; perfect: %.2f%%" % (100 * perfect_rate),
        )
    return mean, sem, perfect_rate, scores, games

def load_agents(
    weight_files,
    overwrite=None,
    device="cuda:0",
    belief_stats=False,
    partner_models_path="None",
    sad_legacy=[0, 0],
):
    agents = []
    sad = []
    hide_action = []
    parameterized = []
    belief_model_path="None"
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = device
    overwrite["boltzmann_act"] = False

    # Load models from weight files
    for i, weight_file in enumerate(weight_files):
        assert os.path.exists(weight_file), f"path file not found: {weight_file}"

        if sad_legacy[i]:
            agent = utils.load_sad_model(weight_file, device)
            agents.append(agent)
            sad.append(True)
            hide_action.append(False)
            parameterized.append(False)
            continue

        try: 
            state_dict = torch.load(weight_file)
        except:
            sys.exit(f"weight_file {weight_file} can't be loaded")

        if "fc_v.weight" in state_dict.keys():
            agent, cfg = utils.load_agent(weight_file, overwrite)
            agents.append(agent)
            sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
            hide_action.append(bool(cfg["hide_action"]))
            if belief_stats:
                belief_model_path = cfg["belief_model"]
            parameterized.append(cfg["parameterized"])
        else:
            agent = utils.load_supervised_agent(weight_file, "cuda:0")
            agents.append(agent)
            sad.append(False)
            hide_action.append(False)
            parameterized.append(False)
        agent.train(False)

    return {
        "agents": agents, 
        "sad": sad,
        "hide_action": hide_action,
        "parameterized": parameterized,
        "belief_model_path": belief_model_path,
    }

def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def load_partner_agents(
        partner_models, 
        partner_type,
        partner_sad_legacy,
    ):
    if partner_models is "None" or len(partner_models) == 0:
        return None

    print(f"loading {partner_type} agents")

    partners = []

    for partner_model_path in partner_models:
        partner_cfg = {
            "sad": False, 
            "hide_action": False,
            "weight": partner_model_path,
            "sad_legacy": False,
        }

        overwrite = {}
        overwrite["vdn"] = False
        overwrite["device"] = "cuda:0"
        overwrite["boltzmann_act"] = False

        if partner_sad_legacy:
            partner_cfg["agent"] = utils.load_sad_model(
                    partner_model_path, "cuda:0")
            partner_cfg["sad"] = True
            partner_cfg["hide_action"] = False
            partner_cfg["parameterized"] = False
            partner_cfg["sad_legacy"] = True
            partners.append(partner_cfg)
            continue

        try: 
            state_dict = torch.load(partner_model_path)
        except:
            sys.exit(f"weight_file {partner_model_path} can't be loaded")

        if "fc_v.weight" in state_dict.keys():
            partner_agent, cfg = utils.load_agent(
                    partner_model_path, overwrite)
            partner_cfg["sad"] = cfg["sad"] if "sad" in cfg else cfg["greedy_extra"]
            partner_cfg["hide_action"] = bool(cfg["hide_action"])
            partner_cfg["parameterized"] = bool(cfg["parameterized"])
            partner_cfg["weight"] = partner_model_path
        else:
            partner_agent = utils.load_supervised_agent(
                    partner_model_path, "cuda:0")
            partner_cfg["sad"] = False
            partner_cfg["hide_action"] = False
            partner_cfg["parameterized"] = False
        partner_cfg["agent"] = partner_agent

        partners.append(partner_cfg)

    return partners
