// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>
#include <iostream> 
#include "rela/tensor_dict.h"
#include "searchcc/game_sim.h"
#include "rela/prioritized_replay.h"
#include "rela/r2d2.h"

namespace search {

inline void addHid(rela::TensorDict& input, const rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto ret = input.emplace(kv.first, kv.second);
    assert(ret.second); }
}

inline void updateHid(rela::TensorDict& output, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto it = output.find(kv.first);
    assert(it != output.end());
    auto newHid = it->second;
    assert(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    output.erase(it);
  }
}

class HybridModel {
 public:
  HybridModel(
      int index, 
      std::vector<bool> legacySad,
      bool legacySadTestPartner,
      std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer,
      bool testPartner,
      int bpIndex,
      bool sba,
      std::vector<std::vector<int>> colourPermute,
      std::vector<std::vector<int>> inverseColourPermute)
      : index(index)
      , rlStep_(0) 
      , legacySad_(legacySad)
      , legacySadTestPartner_(legacySadTestPartner)
      , replayBuffer_(std::move(replayBuffer))
      , testPartner_(testPartner) 
      , bpIndex_(bpIndex) 
      , sba_(sba) 
      , colourPermute_(colourPermute) 
      , inverseColourPermute_(inverseColourPermute) 
      , cpIndex_(0) {
    if (testPartner_) {
      r2d2Buffer_ = std::make_shared<rela::R2D2Buffer>(
            legacySadTestPartner ? 1 : 3, 80, 0.999);
    } else {
      r2d2Buffer_ = std::make_shared<rela::R2D2Buffer>(
            legacySad.at(0) ? 1 : 3, 80, 0.999);
    }
  }

  HybridModel(const HybridModel& m, int bpIndex=0, int cpIndex=0)
      : index(m.index)
      , bpPartnerModel_(m.bpPartnerModel_)
      , bpPartnerHid_(m.bpPartnerHid_)
      , rlModel_(m.rlModel_)
      , rlHid_(m.rlHid_)
      , rlStep_(m.rlStep_) 
      , legacySad_(m.legacySad_) 
      , legacySadTestPartner_(m.legacySadTestPartner_) 
      , replayBuffer_(m.replayBuffer_) 
      , r2d2Buffer_(m.r2d2Buffer_) 
      , testPartner_(m.testPartner_) 
      , bpIndex_(0) 
      , sba_(m.sba_) 
      , colourPermute_(m.colourPermute_) 
      , inverseColourPermute_(m.inverseColourPermute_) 
      , cpIndex_(0) {
    if (m.bpModel_.size() > 0) {
      bpModel_ = std::vector<std::shared_ptr<rela::BatchRunner>>(
                 1, m.bpModel_.at(bpIndex));
    }

    if (m.bpHid_.size() > 0){
      bpHid_ = std::vector<std::vector<rela::TensorDict>>(1, 
          m.bpHid_.at(bpIndex));
    }
    
    if (m.colourPermute_.size() > 0){
      colourPermute_ = std::vector<std::vector<int>>(1, 
          m.colourPermute_.at(cpIndex));
    }

    if (m.inverseColourPermute_.size() > 0){
      inverseColourPermute_ = std::vector<std::vector<int>>(1, 
          m.inverseColourPermute_.at(cpIndex));
    }

    futBp_ = std::vector<std::vector<rela::Future>>(bpModel_.size(), 
        std::vector<rela::Future>(colourPermute_.size()));
  }

  HybridModel& operator=(const HybridModel& m) {
    assert(index == m.index);
    bpModel_ = m.bpModel_;
    bpHid_ = m.bpHid_;
    bpPartnerModel_ = m.bpPartnerModel_;
    bpPartnerHid_ = m.bpPartnerHid_;
    rlModel_ = m.rlModel_;
    rlHid_ = m.rlHid_;
    rlStep_ = m.rlStep_;
    legacySad_ = m.legacySad_;
    legacySadTestPartner_ = m.legacySadTestPartner_;
    replayBuffer_ = m.replayBuffer_;
    r2d2Buffer_ = m.r2d2Buffer_;
    testPartner_ = m.testPartner_;
    bpIndex_ = m.bpIndex_;
    return *this;
  }

  void setBpModel(std::vector<std::shared_ptr<rela::BatchRunner>> bpModel) {
    bpModel_ = bpModel;
    futBp_ = std::vector<std::vector<rela::Future>>(bpModel.size(),
        std::vector<rela::Future>(colourPermute_.size()));
  }

  void setBpModel(std::vector<std::shared_ptr<rela::BatchRunner>> bpModel, 
      std::vector<std::vector<rela::TensorDict>> bpHid) {
    bpModel_ = bpModel;
    bpHid_ = bpHid;
    futBp_ = std::vector<std::vector<rela::Future>>(bpModel.size(),
        std::vector<rela::Future>(colourPermute_.size()));
  }

  void setBpPartnerModel(std::shared_ptr<rela::BatchRunner> bpPartnerModel) {
    bpPartnerModel_ = bpPartnerModel;
  }

  void setBpPartnerModel(std::shared_ptr<rela::BatchRunner> bpPartnerModel, 
      rela::TensorDict bpPartnerHid) {
    bpPartnerModel_ = bpPartnerModel;
    bpPartnerHid_ = bpPartnerHid;
  }

  void setRlModel(std::shared_ptr<rela::BatchRunner> rlModel, rela::TensorDict rlHid) {
    rlModel_ = rlModel;
    rlHid_ = rlHid;
  }

  rela::Future asyncComputeAction(const GameSimulator& env) const;

  // compute bootstrap target/value using blueprint
  rela::Future asyncComputeTarget(
      const GameSimulator& env, float reward, bool terminal) const;

  // compute priority with rl model
  rela::Future asyncComputePriority(const rela::TensorDict& input) const;

  void initialise(bool testActing = false);

  // observe before act
  void observeBeforeAct(
      const GameSimulator& env, 
      float eps, 
      bool testAct = false, 
      rela::TensorDict* retFeat = nullptr);

  int decideAction(
      const GameSimulator& env, 
      bool testActing = false,
      rela::TensorDict* retAction = nullptr);

  int getRlStep() const {
    return rlStep_;
  }

  void setRlStep(int rlStep) {
    assert(rlModel_ != nullptr);
    rlStep_ = rlStep;
  }

  const std::vector<std::vector<rela::TensorDict>>& getBpHid() const {
    return bpHid_;
  }

  void setBpHid(const std::vector<std::vector<rela::TensorDict>>& bpHid) {
    bpHid_ = bpHid;
  }

  void setBeliefHid(const rela::TensorDict& belief_h0) {
    belief_h0_ = belief_h0;
  }

  const rela::TensorDict& getBeliefHid() const {
    return belief_h0_;
  }

  const rela::TensorDict& getBpPartnerHid() const {
    return bpPartnerHid_;
  }

  void setBpPartnerHid(const rela::TensorDict& bpPartnerHid) {
    bpPartnerHid_ = bpPartnerHid;
  }

  void setTestPartnerBeliefHid(const rela::TensorDict& partner_belief_h0) {
    partner_belief_h0_ = partner_belief_h0;
  }

  const rela::TensorDict& getTestPartnerBeliefHid() const {
    return partner_belief_h0_;
  }

  const rela::TensorDict& getRlHid() const {
    return rlHid_;
  }

  void setRlHid(const rela::TensorDict& rlHid) {
    rlHid_ = rlHid;
  }

  void observeAfterAct(const GameSimulator& env, bool testActing = false);
  void pushEpisodeToReplayBuffer();

  std::unordered_map<std::string, std::string> getChosenMoves() {
    return chosenMoves_;
  }

  int getNumBpModels() const {
    return (int)bpModel_.size();
  }

  int getNumBpHid() {
    return (int)bpHid_.size();
  }

  int getBpIndex() {
    return bpIndex_;
  }

  void setBpIndex(int bpIndex) {
    bpIndex_ = bpIndex;
  }

  std::vector<std::vector<rela::TensorDict>> getBpHid() {
    return bpHid_;
  }

  const std::vector<std::vector<int>>& getColourPermute() const {
    return colourPermute_;
  }

  int getNumColourPermute() {
    return (int)colourPermute_.size();
  }

  const std::vector<std::vector<int>>& getInverseColourPermute() const {
    return inverseColourPermute_;
  }

  void setColourPermute(const std::vector<std::vector<int>>& colourPermute) {
    colourPermute_ = colourPermute;
  }

  void setInverseColourPermute(const std::vector<std::vector<int>>& inverseColourPermute) {
    inverseColourPermute_ = inverseColourPermute;
  }

  std::vector<std::shared_ptr<rela::BatchRunner>> getBpModel() {
    return bpModel_;
  }

  const bool hideAction = false;
  const int index;

 private:
  rela::TensorDict observeBp(const GameSimulator& env, bool testActing, 
      int bpIndex, int cpIndex, rela::TensorDict* retFeat = nullptr);

  rela::TensorDict getBpReply(const GameSimulator& env);

  int getRlAction(const GameSimulator& env, rela::TensorDict rlReply, 
      rela::TensorDict bpReply, bool testActing, rela::TensorDict* retAction);

  int getBpAction(const GameSimulator& env, rela::TensorDict bpReply, 
      rela::TensorDict bpPartnerReply, bool testActing, 
      rela::TensorDict* retAction);

  std::vector<std::shared_ptr<rela::BatchRunner>> bpModel_;
    //std::vector<std::shared_ptr<rela::BatchRunner>>(1, 
        //std::shared_ptr<rela::BatchRunner>());
  std::vector<std::vector<rela::TensorDict>> bpHid_;
    //std::vector<std::vector<rela::TensorDict>>(1, 
        //std::vector<rela::TensorDict>(1, rela::TensorDict()));
  rela::TensorDict belief_h0_;
  std::vector<std::vector<rela::Future>> futBp_;
    //std::vector<std::vector<rela::Future>>(1, 
        //std::vector<rela::Future>(1, rela::Future()));

  std::shared_ptr<rela::BatchRunner> bpPartnerModel_;
  rela::TensorDict bpPartnerHid_;
  rela::TensorDict partner_belief_h0_;
  rela::Future futBpPartner_;

  std::shared_ptr<rela::BatchRunner> rlModel_;
  rela::TensorDict rlHid_;
  rela::Future futRl_;

  int rlStep_;

  std::vector<bool> legacySad_;
  bool legacySadTestPartner_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;
  std::shared_ptr<rela::R2D2Buffer> r2d2Buffer_;
  bool testPartner_;
  std::unordered_map<std::string, std::string> chosenMoves_;
  int bpIndex_;
  bool sba_;
  std::vector<std::vector<int>> colourPermute_;
    //std::vector<std::vector<int>>(1, std::vector<int>());
  std::vector<std::vector<int>> inverseColourPermute_;
    //std::vector<std::vector<int>>(1, std::vector<int>());
  int cpIndex_;
};
}  // namespace search
