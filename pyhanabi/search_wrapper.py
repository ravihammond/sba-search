import os
import sys
import pprint
pprint = pprint.pprint
import random

# c++ backend
import set_path
set_path.append_sys_path()
import rela
import hanalearn

import utils
import belief_model

class SearchWrapper:
    def __init__(
        self,
        player_idx,
        public_belief,
        bp_weight_files,
        bp_sad_legacy,
        test_partner_weight_file,
        test_partner_sad_legacy,
        rl_weight_file,
        rl_sad_legacy,
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
        replay_buffer=None,
        is_test_partner=1,
        sba=0,
        colour_permute=None,
        inverse_colour_permute=None,
    ):
        self.player_idx = player_idx
        self.public_belief = public_belief
        assert not public_belief
        self.num_thread = num_thread
        self.num_game_per_thread = num_game_per_thread
        self.num_samples = num_samples
        self.acceptance_rate = 0.05
        self.replay_buffer = replay_buffer
        self.is_test_partner = is_test_partner
        self.explore_eps = explore_eps
        self.gamma = gamma
        self.n_step = n_step
        self.actor = None
        self.train_device = train_device
        self.rollout_bsize = rollout_bsize
        self.log_bsize_freq = log_bsize_freq
        self.bp_rollout_device = bp_rollout_device
        self.bp_sad_legacy = bp_sad_legacy
        self.test_partner_sad_legacy = test_partner_sad_legacy
        self.sba = sba
        self.colour_permute = colour_permute
        self.inverse_colour_permute = inverse_colour_permute

        self.setup_bp(bp_weight_files, bp_sad_legacy, bp_rollout_device)
        self.setup_rl(rl_weight_file, rl_sad_legacy, train_device, rl_rollout_device)
        self.setup_test_partner(test_partner_weight_file, test_partner_sad_legacy,
                bp_rollout_device)
        self.setup_belief(belief_file, belief_device, num_samples)
        self.reset()

    def setup_bp(self, bp_weight_files, bp_sad_legacy, bp_rollout_device):
        print("bp")
        self.bp = []
        self.bp_runner = []

        for weight_file, sad_legacy in zip(bp_weight_files, bp_sad_legacy):
            bp = self.load_model(weight_file, sad_legacy, 1, bp_rollout_device)

            runner = rela.BatchRunner(
                bp,
                bp_rollout_device,
                self.rollout_bsize,
                ["act", "compute_target"],
            )

            if self.log_bsize_freq > 0:
                self.bp_runner.set_log_freq(self.log_bsize_freq)
            runner.start()

            self.bp.append(bp)
            self.bp_runner.append(runner)

    def setup_rl(self, rl_weight_file, rl_sad_legacy, 
            train_device, rl_rollout_device):
        if rl_rollout_device is None:
            self.rl = None
            self.rl_runner = None
            return
        print("rl")

        self.rl = self.load_model(rl_weight_file, rl_sad_legacy, 1, train_device)

        self.rl_runner = rela.BatchRunner(
            self.rl.clone(rl_rollout_device),
            rl_rollout_device,
            self.rollout_bsize,
            ["act", "compute_priority"],
        )

        self.rl_runner.start()

    def setup_test_partner(self, test_partner_weight_file, 
            test_partner_sad_legacy, bp_rollout_device):
        if not self.is_test_partner:
            self.test_partner = None
            self.test_partner_runner = None
            return
        print("test partner")

        self.test_partner = self.load_model(
                test_partner_weight_file, test_partner_sad_legacy, 
                1, bp_rollout_device)

        self.test_partner_runner = rela.BatchRunner(
            self.test_partner,
            bp_rollout_device,
            self.rollout_bsize,
            ["act", "compute_target"],
        )

        if self.log_bsize_freq > 0:
            self.test_partner_runner.set_log_freq(self.log_bsize_freq)

        self.test_partner_runner.start()

    def load_model(self, weight_file, sad, multi_step, model_device):
        if sad:
            model = utils.load_sad_model(
                    weight_file, 
                    model_device,
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
            model, config = utils.load_agent(
                weight_file, {
                    "device": model_device, 
                    "off_belief": False,
                    "multi_step": 1
                }
            )

        assert not config["hide_action"]
        assert not config["boltzmann_act"]
        assert config["method"] == "iql"
        assert model.multi_step == 1

        return model

    def setup_belief(self, belief_file, belief_device, num_samples):
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
                self.belief_model, belief_device, 
                self.rollout_bsize, ["observe", "sample"]
            )
            self.belief_runner.start()
        else:
            self.belief_runner = None

    def reset(self):
        self.actor = hanalearn.RLSearchActor(
            self.player_idx,
            self.bp_runner,
            self.test_partner_runner,
            self.rl_runner,
            self.belief_runner,  # belief runner
            self.num_samples,  # num samples
            self.public_belief,  # public belief
            False,  # joint search
            self.explore_eps,
            self.n_step,
            self.gamma,
            random.randint(1, 999999),
            self.bp_sad_legacy,
            self.test_partner_sad_legacy,
            self.replay_buffer,
            self.is_test_partner,
            -1 if self.is_test_partner else 0,
            self.sba,
            self.colour_permute,
            self.inverse_colour_permute,
        )
        self.actor.set_compute_config(self.num_thread, self.num_game_per_thread)

    def update_rl_model(self, model):
        self.rl_runner.update_model(model)

    def reset_rl_to_bp(self):
        self.rl.online_net.load_state_dict(self.bp[0].online_net.state_dict())
        self.rl.target_net.load_state_dict(self.bp[0].online_net.state_dict())
        self.update_rl_model(self.rl)
        self.actor.reset_rl_rnn()

