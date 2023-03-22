// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>
#include <iostream>

#include "rela/thread_loop.h"
#include "rlcc/r2d2_actor.h"

#define PR true

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::vector<std::shared_ptr<HanabiEnv>> envs,
      std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors,
      bool eval)
      : envs_(std::move(envs))
      , actors_(std::move(actors))
      , done_(envs_.size(), -1)
      , eval_(eval) {
    assert(envs_.size() == actors_.size());
  }

  virtual void mainLoop() override {
    int game_turn = 0;
    while (!terminated()) {
      if(PR)printf("\n=======================================\n");
      if(PR)printf("Game Turn: %d\n", game_turn);
      // go over each envs in sequential order
      // call in seperate for-loops to maximize parallization
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];

        if (envs_[i]->terminated()) {
          // we only run 1 game for evaluation
          if (eval_) {
            ++done_[i];
            if (done_[i] == 1) {
              numDone_ += 1;
              if (numDone_ == (int)envs_.size()) {
                return;
              }
            }
          }

          envs_[i]->reset();
          for (size_t j = 0; j < actors.size(); ++j) {
            if(PR)printf("\n[player %ld resetting]\n", j);
            actors[j]->reset(*envs_[i]);
          }
        }
      }

      if(PR)printf("\n%s\n", envs_[0]->getHleState().ToString().c_str());

      if(PR)printf("\n----\n");

      for (size_t i = 0; i < envs_.size(); ++i) {
        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR) printf("\n[player %ld observe before acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->observeBeforeAct(*envs_[i]);
        }
      }

      if(PR)printf("\n----\n");

      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld acting]%s\n", j, 
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->act(*envs_[i], curPlayer);
        }
      }

      if(PR)printf("\n----\n");

      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld fictious acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->fictAct(*envs_[i]);
        }
      }

      if(PR)printf("\n----\n");

      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld observe after acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->observeAfterAct(*envs_[i]);
        }
      }

      game_turn = envs_[0]->numStep();
    }
  }

 private:
  std::vector<std::shared_ptr<HanabiEnv>> envs_;
  std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors_;
  std::vector<int8_t> done_;
  const bool eval_;
  int numDone_ = 0;
};
