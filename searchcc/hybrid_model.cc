// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include "searchcc/hybrid_model.h"

#define PR true

namespace search {

rela::Future HybridModel::asyncComputeAction(const GameSimulator& env) const {
  auto input = observe(env.state(), index, hideAction, legacySad_[0]);
  input["actor_index"] = torch::tensor(index);
  if (rlStep_ > 0) {
    addHid(input, rlHid_);
    input["eps"] = torch::tensor(std::vector<float>{0});
    return rlModel_->call("act", input);
  } else {
    addHid(input, bpHid_[bpIndex_][cpIndex_]);
    return bpModel_[bpIndex_]->call("act", input);
  }
}

// compute bootstrap target/value using blueprint
rela::Future HybridModel::asyncComputeTarget(
    const GameSimulator& env, float reward, bool terminal) const {
  auto feat = observe(env.state(), index, false, legacySad_[0]);
  feat["reward"] = torch::tensor(reward);
  feat["terminal"] = torch::tensor((float)terminal);
  addHid(feat, bpHid_[bpIndex_][cpIndex_]);
  return bpModel_[bpIndex_]->call("compute_target", feat);
}

// compute priority with rl model
rela::Future HybridModel::asyncComputePriority(const rela::TensorDict& input) const {
  assert(rlModel_ != nullptr);
  return rlModel_->call("compute_priority", input);
}

void HybridModel::initialise(bool testActing) {
  if (testActing) {
    rela::TensorDict hid;
    if (testPartner_) {
      hid = bpPartnerHid_;
    } else if (bpIndex_ != -1) {
      hid = bpHid_[bpIndex_][cpIndex_];
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

  // Observe for all bp models
  if(PR)printf("bp calling act\n");
  rela::TensorDict feat;
  if (bpIndex_ == -1) {
    for (int i = 0; i < (int)bpModel_.size(); i++) {
      for (int j = 0; j < (int)colourPermute_.size(); j++) {
        observeBp(env, testActing, i, j);
      }
    }
  } else { 
    feat = observeBp(env, testActing, bpIndex_, 0, retFeat);
  }

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

rela::TensorDict HybridModel::observeBp(
    const GameSimulator& env, 
    bool testActing, 
    int bpIndex, 
    int cpIndex, 
    rela::TensorDict* retFeat) {
  auto feat = observe(env.state(), index, hideAction, legacySad_.at(bpIndex),
      sba_, colourPermute_.at(cpIndex), inverseColourPermute_.at(cpIndex));

  if (retFeat != nullptr) {
    *retFeat = feat;
  }

  auto input = feat;
  input["actor_index"] = torch::tensor(index);

  if (testActing && replayBuffer_ != nullptr && !testPartner_) {
    r2d2Buffer_->pushObs(input);
  }

  addHid(input, bpHid_.at(bpIndex).at(cpIndex));
  std::stringstream permute;
  std::copy(colourPermute_.at(cpIndex).begin(), colourPermute_.at(cpIndex).end(), 
            std::ostream_iterator<int>(permute, " "));
  if (sba_) {
    if(PR)printf("[ %s], ", permute.str().c_str());
  }
  if(PR)bpModel_.at(bpIndex)->printModel();
  futBp_.at(bpIndex).at(cpIndex) = bpModel_.at(bpIndex)->call("act", input);

  return feat;
}

int HybridModel::decideAction(
    const GameSimulator& env, 
    bool testActing, 
    rela::TensorDict* retAction) {
  // Get bp results, and update hid
  int action = -1;
  rela::TensorDict rlReply;
  rela::TensorDict bpPartnerReply;
  rela::TensorDict bpReply = getBpReply(env);

  // Get partner bp results, and update hid
  if (testActing && testPartner_) {
    bpPartnerReply = futBpPartner_.get();
    updateHid(bpPartnerReply, bpPartnerHid_);
  }

  // Record RL Search actor stats
  if (rlStep_ > 0 || (testActing && !testPartner_)) {
    rlReply = futRl_.get();
    int rlAction = rlReply.at("a").item<int64_t>();
    auto move = env.state().ParentGame()->GetMove(rlAction);
    if (testActing) {
      chosenMoves_["rl action"] = move.ToString();
      if (replayBuffer_ != nullptr) {
        for (auto& kv: rlReply) {
          // Only store stats for actions and q values
          if (kv.first != "a" && kv.first != "all_q") {
            continue;
          }
          std::string newKey = "rl_" + kv.first;
          bpReply.at(newKey) = kv.second;
        }
        bpReply["rl_actor"] = torch::tensor(1);
      }
    }
  }

  // Record blueprint actor stats.
  if (testActing && testPartner_ && replayBuffer_ != nullptr) {
    for (auto& kv: bpPartnerReply) {
      // Only store stats for actions and q values
      if (kv.first != "a" && kv.first != "all_q") {
        continue;
      }
      std::string newKey = "rl_" + kv.first;
      bpPartnerReply.at(newKey) = torch::zeros(kv.second.sizes());
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

  // RL action will be used
  if (rlStep_ > 0) {
    action = getRlAction(env, rlReply, bpReply, testActing, retAction);
  // Blueprint action will be used
  } else {
    assert(futRl_.isNull());
    action = getBpAction(env, bpReply, bpPartnerReply, testActing, retAction);
    auto actionStr = env.game().GetMove(action).ToString();
  }

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }

  return action;
}

rela::TensorDict HybridModel::getBpReply(const GameSimulator& env) {
  char colourMap[5] = {'R', 'Y', 'G', 'W', 'B'};
  rela::TensorDict bpReply;

  if (bpIndex_ >= 0) {
    bpReply = futBp_.at(bpIndex_).at(cpIndex_).get();
    updateHid(bpReply, bpHid_.at(bpIndex_).at(cpIndex_));
    return bpReply;
  }

  for (int i = 0; i < (int)bpHid_.size(); i++) {
    for (int j = 0; j < (int)colourPermute_.size(); j++) {
      auto reply = futBp_.at(i).at(j).get();
      updateHid(reply, bpHid_.at(i).at(j));

      // Print permutes, and model name
      std::stringstream permute;
      std::copy(colourPermute_.at(j).begin(), colourPermute_.at(j).end(), 
      std::ostream_iterator<int>(permute, " "));
      if (sba_) {
        if(PR)printf("[ %s], ", permute.str().c_str());
      }
      if(PR)bpModel_.at(i)->printModel();

      // Print action, and model name
      int action = reply.at("a").item<int64_t>();
      auto move = env.state().ParentGame()->GetMove(action);
      if (sba_ && move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
        char colourBefore = colourMap[move.Color()];
        int realColor = inverseColourPermute_.at(j).at(move.Color());
        move.SetColor(realColor);
        if(PR)printf("action colour %c->%c, %s\n", 
            colourBefore, colourMap[move.Color()], move.ToString().c_str());
      }
    }
  }

  return rela::TensorDict();
}

int HybridModel::getRlAction(
    const GameSimulator& env, 
    rela::TensorDict rlReply, 
    rela::TensorDict bpReply, 
    bool testActing, 
    rela::TensorDict* retAction) {
  updateHid(rlReply, rlHid_);
  int action = rlReply.at("a").item<int64_t>();

  if (env.state().CurPlayer() == index) {
    --rlStep_;
  }

  // Save bp action for logs
  if (testActing) {
    int bpAction = bpReply.at("a").item<int64_t>();
    auto bpMove = env.state().ParentGame()->GetMove(bpAction);
    chosenMoves_["bp action"] = bpMove.ToString();
  }

  if (retAction != nullptr) {
    *retAction = rlReply;
  }

  return action;
}

int HybridModel::getBpAction(
    const GameSimulator& env, 
    rela::TensorDict bpReply, 
    rela::TensorDict bpPartnerReply,
    bool testActing, 
    rela::TensorDict* retAction) {
  int action = -1;

  // Use the test partner blueprint action for the real game.
  if (testActing && testPartner_) {
    action = bpPartnerReply.at("a").item<int64_t>();
    auto move = env.state().ParentGame()->GetMove(action);
    chosenMoves_["bp partner action"] = move.ToString();

  // Use the blueprint action when not test partner, or when in eval 
  // or search mode.
  } else {
    action = bpReply.at("a").item<int64_t>();
    auto move = env.state().ParentGame()->GetMove(action);

    // un-shuffle colour action for test partner.
    if (testPartner_ && sba_ && 
        move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      int realColor = inverseColourPermute_[cpIndex_][move.Color()];
      move.SetColor(realColor);
      action = env.game().GetMoveUid(move);
    }

    chosenMoves_["bp action"] = move.ToString();
  }

  // assert(retAction == nullptr);
  // technically this is not right, we should never return action from bp
  // for training purpose, but in this case it will be skip anyway.
  if (retAction != nullptr) {
    if (sba_) {
      printf("returning action, actor 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    }
    assert(action == env.game().MaxMoves());
    *retAction = bpReply;
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

