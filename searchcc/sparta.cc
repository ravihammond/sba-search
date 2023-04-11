// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <chrono>
#include <future>

#include "searchcc/sparta.h"

#define SIM false

namespace search {

float searchMove(
    const hle::HanabiState& state,
    hle::HanabiMove move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int> seeds,
    int myIdx,
    const std::vector<HybridModel>& players,
    bool skipSearch) {
  if (skipSearch) return 0;
  //printf("searchMove() sparta.cc ======\n");
  std::vector<std::vector<HybridModel>> allPlayers(hands.size(), players);
  std::vector<GameSimulator> games;
  games.reserve(hands.size());
  //printf("num games before: %d\n", (int)games.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    std::vector<SimHand> simHands{
        SimHand(myIdx, hands[i]),
    };
    games.emplace_back(state, simHands, seeds[i]);
  }

  //printf("num games after: %d\n", (int)games.size());

  size_t terminated = 0;
  std::vector<int> notTerminated;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(!games[i].terminal());
    games[i].step(move);
    if (!games[i].terminal()) {
      notTerminated.push_back(i);
    } else {
      ++terminated;
    }
  }

  int game_turn = 0;
  while (!notTerminated.empty()) {
    if(SIM)printf("\n=======================================\n");
    if(SIM)printf("Game Turn: %d\n", game_turn);
    if(SIM)printf("\n%s\n", games[0].state().ToString().c_str());

    std::vector<int> newNotTerminated;
    for (auto i : notTerminated) {
      assert(!games[i].state().IsTerminal());
      for (auto& actor : allPlayers[i]) {
        actor.observeBeforeAct(games[i], 0);
      }
    }

    for (auto i : notTerminated) {
      auto& game = games[i];
      int action = -1;
      for (auto& actor : allPlayers[i]) {
        int a = actor.decideAction(game, false);
        if (actor.index == game.state().CurPlayer()) {
          action = a;
        }
      }
      // std::cout << "move: " << game.getMove(action).ToString() << std::endl;
      game.step(game.getMove(action));
      if (!game.terminal()) {
        newNotTerminated.push_back(i);
      } else {
        ++terminated;
      }
    }

    notTerminated = newNotTerminated;
    game_turn++;
  }
  assert(terminated == games.size());

  std::vector<float> scores(games.size());
  float mean = 0;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(games[i].terminal());
    scores[i] = games[i].score();
    mean += scores[i];
  }
  mean = mean / scores.size();
  return mean;
}

// should be called after decideAction?
hle::HanabiMove SpartaActor::spartaSearch(
    const GameSimulator& env, hle::HanabiMove bpMove, int numSearch, float threshold) {
  torch::NoGradGuard ng;
  printf("spartaSearch() sparta.cc ======\n");

  const auto& state = env.state();
  assert(state.CurPlayer() == index);
  auto legalMoves = state.LegalMoves(index);
  printf("num legal moves: %d\n", (int)legalMoves.size());
  printf("num search: %d\n", numSearch);

  if (legalMoves.empty()) {
    return bpMove;
  }

  int numSearchPerMove = numSearch / legalMoves.size();
  std::cout << "SPARTA will run " << numSearchPerMove << " searches on "
            << legalMoves.size() << " legal moves" << std::endl;
  auto simHands = handDist_.sampleHands(numSearchPerMove, &rng_);
  printf("num sim hands: %d\n", (int)simHands.size());

  std::vector<int> seeds;
  for (size_t i = 0; i < simHands.size(); ++i) {
    seeds.push_back(int(rng_()));
  }

  std::vector<std::future<float>> futMoveScores;
  std::vector<HybridModel> players;
  for (auto& p : partners_) {
    players.push_back(p->model_);
  }
  for (int i = 0; i < (int)legalMoves.size(); i++) {
    auto move = legalMoves.at(i);
    bool skipSearch = false;
    //if (i == 0) skipSearch = false;
    futMoveScores.push_back(std::async(std::launch::async, searchMove, 
        state, 
        move, 
        simHands, 
        seeds, 
        index, 
        players,
        skipSearch));
  }

  hle::HanabiMove bestMove = bpMove;
  float bpScore = -1;
  float bestScore = -1;

  std::cout << "SPARTA scores for moves:" << std::endl;
  for (size_t i = 0; i < legalMoves.size(); ++i) {
    float score = futMoveScores[i].get();
    auto move = legalMoves[i];
    if (move == bpMove) {
      assert(bpScore == -1);
      bpScore = score;
    }
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    std::cout << move.ToString() << ": " << score << std::endl;
  }

  std::cout << "SPARTA best - bp: " << bestScore - bpScore << std::endl;
  if (bestScore - bpScore >= threshold) {
    std::cout << "SPARTA changes move from " << bpMove.ToString() << " to "
              << bestMove.ToString() << std::endl;
    return bestMove;
  } else {
    return bpMove;
  }
}

}  // namespace search
