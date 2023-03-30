// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h>
#include <iostream>
#include "searchcc/hybrid_model.h"

#define PR false

namespace search {

rela::Future HybridModel::asyncComputeAction(const GameSimulator& env) const {
  auto input = observe(env.state(), index, hideAction, legacySad_[0]);
  input["actor_index"] = torch::tensor(index);
  if (rlStep_ > 0) {
    addHid(input, rlHid_);
    input["eps"] = torch::tensor(std::vector<float>{0});
    return rlModel_->call("act", input);
  } else {
    addHid(input, bpHid_[0]);
    return bpModel_[0]->call("act", input);
  }
}

// compute bootstrap target/value using blueprint
rela::Future HybridModel::asyncComputeTarget(
    const GameSimulator& env, float reward, bool terminal) const {
  auto feat = observe(env.state(), index, false, legacySad_[0]);
  feat["reward"] = torch::tensor(reward);
  feat["terminal"] = torch::tensor((float)terminal);
  addHid(feat, bpHid_[0]);
  return bpModel_[0]->call("compute_target", feat);
}

// compute priority with rl model
rela::Future HybridModel::asyncComputePriority(const rela::TensorDict& input) const {
  assert(rlModel_ != nullptr);
  return rlModel_->call("compute_priority", input);
}

void HybridModel::initialise(bool testActing) {
  if (testActing) {
    auto hid = bpHid_[0];

    if (testPartner_) {
      hid = bpPartnerHid_;
    }

    if (r2d2Buffer_ != nullptr) {
      r2d2Buffer_->init(hid);
    }
  }
}

// observe before act
void HybridModel::observeBeforeAct(
    const GameSimulator& env, 
    float eps, 
    bool testActing, 
    rela::TensorDict* retFeat) {
  chosenMoves_.clear();
  auto feat = observe(env.state(), index, hideAction, legacySad_[0]);
  if (retFeat != nullptr) {
    *retFeat = feat;
  }
  auto input = feat;
  input["actor_index"] = torch::tensor(index);
  if (testActing && replayBuffer_ != nullptr && !testPartner_) {
    r2d2Buffer_->pushObs(input);
  }

  addHid(input, bpHid_[0]);
  if(PR)printf("bp calling act\n");
  if(PR)bpModel_[0]->printModel();
  futBp_ = bpModel_[0]->call("act", input);

  // forward bp regardless of whether rl is used

  rela::TensorDict inputPartner;

  if (testActing && testPartner_) {
    auto featPartner = observe(env.state(), index, hideAction, legacySadTestPartner_);

    inputPartner = featPartner;
    inputPartner["actor_index"] = torch::tensor(index);

    if (replayBuffer_ != nullptr && testPartner_) {
      r2d2Buffer_->pushObs(inputPartner);
    }

    addHid(inputPartner, bpPartnerHid_);
    if(PR)printf("bp partner calling act\n");
    if(PR)bpPartnerModel_->printModel();
    futBpPartner_ = bpPartnerModel_->call("act", inputPartner);
  }

  // maybe forward rl
  if (rlStep_ > 0 || (testActing && !testPartner_)) {
    feat["eps"] = torch::tensor(std::vector<float>{eps});
    auto rlInput = feat;
    addHid(rlInput, rlHid_);

    if(PR)printf("rl calling act\n");
    if(PR)rlModel_->printModel();
    futRl_ = rlModel_->call("act", rlInput);
  }
}

int HybridModel::decideAction(
    const GameSimulator& env, 
    bool verbose, 
    bool testActing, 
    rela::TensorDict* retAction) {
  (void)verbose;
  // Get bp results, and update hid
  int action = -1;
  auto bpReply = futBp_.get();
  updateHid(bpReply, bpHid_[0]);

  // Get partner bp results, and update hid
  rela::TensorDict bpPartnerReply;
  if (testActing && testPartner_) {
    bpPartnerReply = futBpPartner_.get();
    updateHid(bpPartnerReply, bpPartnerHid_);
  }

  // Get rl results
  rela::TensorDict rlReply;
  if (rlStep_ > 0 || (testActing && !testPartner_)) {
    rlReply = futRl_.get();
    auto rlaction = rlReply.at("a").item<int64_t>();
    auto move = env.state().ParentGame()->GetMove(rlaction);
    if (testActing) {
      chosenMoves_["rl action"] = move.ToString();

      if (replayBuffer_ != nullptr) {
        for (auto& kv: rlReply) {
        if (kv.first != "a" && kv.first != "all_q") continue;
          std::string newKey = "rl_" + kv.first;
          bpReply[newKey] = kv.second;
        }
        bpReply["rl_actor"] = torch::tensor(1);
      }
    }
  }

  if (testActing && testPartner_ && replayBuffer_ != nullptr) {
    for (auto& kv: bpPartnerReply) {
      if (kv.first != "a" && kv.first != "all_q") continue;
      std::string newKey = "rl_" + kv.first;
      bpPartnerReply[newKey] = torch::zeros(kv.second.sizes());
    }
    bpPartnerReply["rl_actor"] = torch::tensor(0);
  }

  // push action results to buffer
  if (testActing && replayBuffer_ != nullptr) {
    if (testPartner_) {
      r2d2Buffer_->pushAction(bpPartnerReply);
    } else {
      r2d2Buffer_->pushAction(bpReply);
    }
  }

  if (rlStep_ > 0) {
    updateHid(rlReply, rlHid_);

    action = rlReply.at("a").item<int64_t>();

    if (env.state().CurPlayer() == index) {
      --rlStep_;
    }

    if (testActing) {
      int bpAction = bpReply.at("a").item<int64_t>();
      auto bpMove = env.state().ParentGame()->GetMove(bpAction);
      chosenMoves_["bp action"] = bpMove.ToString();
    }

    if (retAction != nullptr) {
      *retAction = rlReply;
    }
  } else {
    assert(futRl_.isNull());
    action = bpReply.at("a").item<int64_t>();
    auto move = env.state().ParentGame()->GetMove(action);
    chosenMoves_["bp action"] = move.ToString();

    if (testActing && testPartner_) {
      action = bpPartnerReply.at("a").item<int64_t>();
      move = env.state().ParentGame()->GetMove(action);
      chosenMoves_["bp partner action"] = move.ToString();
    }

    // assert(retAction == nullptr);
    // technically this is not right, we should never return action from bp
    // for training purpose, but in this case it will be skip anyway.
    if (retAction != nullptr) {
      assert(action == env.game().MaxMoves());
      *retAction = bpReply;
    }
  }

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }

  return action;
}


void HybridModel::observeAfterAct(const GameSimulator& env, bool testActing) {
  if (testActing && replayBuffer_ != nullptr) {
    bool terminated = false;
    if (env.state().IsTerminal()) {
      terminated = true;
    }
    r2d2Buffer_->pushReward(0);
    r2d2Buffer_->pushTerminal(float(terminated));
  }
}

void HybridModel::pushEpisodeToReplayBuffer() {
  torch::NoGradGuard ng;
  if (replayBuffer_ == nullptr) {
    return;
  }

  auto lastEpisode = r2d2Buffer_->popTransition();

  auto fullEpisode = lastEpisode.toDict();

  replayBuffer_->add(std::move(lastEpisode), 0);
}

}  // namespace search

