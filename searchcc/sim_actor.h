// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/prioritized_replay.h"
#include "rela/r2d2.h"
#include "searchcc/hybrid_model.h"

namespace search {

class SimulationActor {
 public:
  // training mode
  SimulationActor(
      const HybridModel& model,
      int numRlStep,
      const std::vector<float>& epsList,
      const std::shared_ptr<rela::RNNPrioritizedReplay>& replay,
      int nStep,
      float gamma,
      int seed,
      std::mt19937& cpRng)
      : model_(model)
      , initBpHid_(model.getBpHid().at(0))
      , initRlHid_(model.getRlHid())
      , initBeliefHid_(model.getBeliefHid())
      , initRlStep_(numRlStep)
      , epsList_(epsList)
      , beliefMode_(false)
      , replayBuffer_(replay)
      , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(nStep, 2 * numRlStep, gamma))
      , rng_(seed) 
      , initColorPermute_(model.getColourPermute()) 
      , initInverseColorPermute_(model.getInverseColourPermute()) 
      , cpRng_(cpRng) {
    // do not train with RL when the model is using RL
    assert(model.getRlStep() == 0);
    assert(replay != nullptr);
    resetEps();

    if (initRlStep_ > 0) {
      model_.setRlStep(initRlStep_);
    }
    r2d2Buffer_->init(initRlHid_);
  }

  // eval mode, evaluate the mode as is
  SimulationActor(const HybridModel& model, int bpIndex, std::mt19937& cpRng)
      : model_(model, bpIndex)
      , initBpHid_(model.getBpHid().at(bpIndex))
      , initRlHid_(model.getRlHid())
      , initBeliefHid_(model.getBeliefHid())
      , initRlStep_(model.getRlStep())
      , epsList_({0.0f})
      , beliefMode_(false)
      , replayBuffer_(nullptr)
      , r2d2Buffer_(nullptr)
      , rng_(1) 
      , bpIndex_(bpIndex) 
      , initColorPermute_(model.getColourPermute()) 
      , initInverseColorPermute_(model.getInverseColourPermute()) 
      , cpRng_(cpRng) {
    resetEps();
    resetBp();
  }

  // belief mode, evaluate the model as is, collect data
  SimulationActor(
      const HybridModel& model,
      const std::shared_ptr<rela::RNNPrioritizedReplay>& replay,
      int nStep,
      std::mt19937& cpRng)
      : model_(model)
      , initBpHid_(model.getBpHid().at(0))
      , initRlHid_(model.getRlHid())
      , initBeliefHid_(model.getBeliefHid())
        // A bit hacky but works for one step training as long as resets are done properly
      , initRlStep_(1)
      , epsList_({0.0f})
      , beliefMode_(true)
      , replayBuffer_(replay)
        // also hard coded for one step training
        // TODO: FIXME, this used to be std::make_unique<rela::R2D2Buffer>(nStep, 2, 1.0, true)
      , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(nStep, 2, 1.0))
      , rng_(1) 
      , initColorPermute_(model.getColourPermute()) 
      , initInverseColorPermute_(model.getInverseColourPermute()) 
      , cpRng_(cpRng) {
    resetEps();
    r2d2Buffer_->init(initBeliefHid_);
    model_.setRlStep(initRlStep_);
  }

  // eval mode, evaluate the model as if it will use RL model
  SimulationActor(
      const HybridModel& model, 
      int numRlStep,
      int bpIndex,
      std::mt19937& cpRng)
      : model_(model, bpIndex)
      , initBpHid_(model.getBpHid().at(bpIndex))
      , initRlHid_(model.getRlHid())
      , initBeliefHid_(model.getBeliefHid())
      , initRlStep_(numRlStep)
      , epsList_({0.0f})
      , beliefMode_(false)
      , replayBuffer_(nullptr)
      , r2d2Buffer_(nullptr)
      , rng_(1)
      , bpIndex_(bpIndex) 
      , initColorPermute_(model.getColourPermute())
      , initInverseColorPermute_(model.getInverseColourPermute()) 
      , cpRng_(cpRng) {
    assert(model_.getRlStep() == 0);
    if (initRlStep_ > 0) {
      model_.setRlStep(initRlStep_);
    }
    resetEps();
    resetBp();
  }

  void reset() {
    resetBp();
    model_.setRlHid(initRlHid_);
    if (initRlStep_ > 0) {
      model_.setRlStep(initRlStep_);
    }

    if (r2d2Buffer_ != nullptr) {
      if (beliefMode_) {
        r2d2Buffer_->init(initBeliefHid_);
      } else {
        r2d2Buffer_->init(initRlHid_);
      }
    }

    numObserve_ = 0;
    resetEps();

    callOrder_ = 0;
    needReset_ = false;
  }

  void resetBp() {
    std::uniform_int_distribution<int> dist(0, initColorPermute_.size() - 1);
    int cpIndex = dist(cpRng_);

    std::stringstream permute;
    std::copy(initColorPermute_[cpIndex].begin(), initColorPermute_[cpIndex].end(), 
              std::ostream_iterator<int>(permute, " "));
    printf("colour permute index: %d [ %s]\n", cpIndex, permute.str().c_str());


    model_.setBpHid(std::vector<std::vector<rela::TensorDict>>(1, 
          std::vector<rela::TensorDict>(1, initBpHid_[cpIndex])));
    model_.setColourPermute(std::vector<std::vector<int>>(1, 
          initColorPermute_[cpIndex]));
    model_.setInverseColourPermute(std::vector<std::vector<int>>(1, 
          initInverseColorPermute_[cpIndex]));
  }

  void observeBeforeAct(const GameSimulator& env);

  int decideAction(const GameSimulator& env);

  void observeAfterAct(const GameSimulator& env);

  bool maybeEndEpisode(const GameSimulator& env);

  int getBpIndex() {
    return model_.getBpIndex();
  }

 private:
  void resetEps() {
    eps_ = epsList_[rng_() % epsList_.size()];
  }

  HybridModel model_;
  const std::vector<rela::TensorDict> initBpHid_;
  const rela::TensorDict initRlHid_;
  const rela::TensorDict initBeliefHid_;
  const int initRlStep_;
  const std::vector<float> epsList_;
  const bool beliefMode_;

  std::shared_ptr<rela::RNNPrioritizedReplay> const replayBuffer_;
  std::unique_ptr<rela::R2D2Buffer> const r2d2Buffer_;
  int numObserve_ = 0;
  float eps_;

  rela::Future futTarget_;
  rela::Future futPriority_;
  rela::RNNTransition lastEpisode_;

  std::mt19937 rng_;

  int callOrder_ = 0;
  bool needReset_ = false;
  int bpIndex_ = 0;
  const std::vector<std::vector<int>> initColorPermute_;
  const std::vector<std::vector<int>> initInverseColorPermute_;
  std::mt19937& cpRng_;
};
}  // namespace search
