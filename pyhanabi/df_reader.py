import pandas as pd
import numpy as np
import argparse
import pprint
pprint = pprint.pprint

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

def main(args):
    df = pd.read_pickle(args.path, compression="gzip")

    for i in range(5):
        df[f"card_{i}"] = CARD_ID_TO_STRING[df[f"card_{i}"]]
        df[f"p_card_{i}"] = CARD_ID_TO_STRING[df[f"partner_card_{i}"]]

    df["action"] = ACTION_ID_TO_STRING_SHORT[df["action"]]
    df["rl_action"] = ACTION_ID_TO_STRING_SHORT[df["rl_action"]]

    df["R_fw"] = df["red_fireworks"]
    df["Y_fw"] = df["yellow_fireworks"]
    df["G_fw"] = df["green_fireworks"]
    df["W_fw"] = df["white_fireworks"]
    df["B_fw"] = df["blue_fireworks"]
    df["rl_chosen"] = df["rl_action_chosen"]
    df["thresh"] = df["diff_threshold"]

    columns = [
        "player",
        "turn",
        "lives",
        "score",
        "card_0",
        "card_1",
        "card_2",
        "card_3",
        "card_4",
        "p_card_0",
        "p_card_1",
        "p_card_2",
        "p_card_3",
        "p_card_4",
        "R_fw",
        "Y_fw",
        "G_fw",
        "W_fw",
        "B_fw",
        "action",
        "rl_action",
        "rl_chosen",
        "rl_score",
        "bp_score",
        "rl_bp_diff",
        "rl_bp_diff",
        "thresh",
        "rl_actor",
    ]

    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)

