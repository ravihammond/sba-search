import os
import sys
import pprint
pprint = pprint.pprint
import numpy as np
import pandas as pd
from datetime import datetime


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


def replay_to_dataframe(args, replay_buffer, now, game_data):
    if replay_buffer is None:
        return
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    batch1, batch2 = replay_buffer.sample_from_list_split(2, "cpu", [0, 1])
    data = batch_to_dataset(args, batch1, batch2, date_time, game_data)

    return data


def batch_to_dataset(args, batch1, batch2, date_time, game_data):
    df = pd.DataFrame()

    obs_df = player_dataframe(args, batch1, 0, date_time, game_data[0])
    df = pd.concat([df, obs_df])

    obs_df = player_dataframe(args, batch2, 1, date_time, game_data[1])
    df = pd.concat([df, obs_df])

    df = df.reset_index(drop=True)

    if args.verbose:
        print("num cows:", df.shape[0])
        print("num columns:", len(list(df.columns.values)))

    columns = [
        "game",
        "player",
        "partner",
        "turn",
        "card_0",
        "card_1",
        "card_2",
        "card_3",
        "card_4",
        "action",
        "red_fireworks",
        "yellow_fireworks",
        "green_fireworks",
        "white_fireworks",
        "blue_fireworks",
        # "rl_action",
        # "rl_action_chosen",
        # "rl_score",
        # "bp_score",
        # "rl_bp_diff",
        # "rl_bp_diff",
        # "diff_threshold",
        # "rl_actor",
    ]

    # pprint(df.columns.tolist())
    # print(df[columns].to_string(index=False))
    # print(df.to_string(index=False))

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
        game_names.append(f"{args.player_name[1]}_vs_{args.player_name[0]}_game_{i}")

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

    if (player == 0 and args.test_partner_sad_legacy) or \
       (player == 1 and args.sad_legacy):
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

