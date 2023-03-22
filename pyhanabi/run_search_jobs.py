import argparse
import subprocess
import asyncio
import sys

async def main(args):
    jobs = create_jobs(args)
    sem = asyncio.Semaphore(args.max_processes)
    await asyncio.gather(*[run_job(job, sem) for job in jobs])

def create_jobs(args):
    seed = args.seed_start
    jobs = []

    for _ in range(args.num_games):
        partner_index = seed % args.num_partners
        cuda_divider = 2 * (seed % 2)
        train_device = cuda_divider
        rollout_device = cuda_divider + 1
        jobs.append(["python","rl_search.py", 
            "--save_dir",f"{args.save_dir}/{args.model}",
            "--weight_file", f"../models/my_models/{args.model}",
            "--game_seed", str(seed),
            "--seed", str(seed),
            "--burn_in_frames", "5000",
            "--replay_buffer_size", "100000",
            "--rl_rollout_device", f"cuda:{rollout_device}",
            "--bp_rollout_device", f"cuda:{rollout_device}",
            "--rollout_batchsize", "8000",
            "--num_thread", "1",
            "--num_game_per_thread", "20",
            "--train_device", f"cuda:{train_device}",
            "--batchsize", "128",
            "--num_epoch", "1",
            "--epoch_len", "5000",
            "--belief_device", f"cuda:{train_device}",
            "--num_samples", "50000",
            "--maintain_exact_belief", "1",
            "--search_exact_belief", "1",
            "--partner_models", "agent_groups/all_sad.json",
            "--partner_sad_legacy", "1",
            "--train_test_splits", "sad_train_test_splits_1_12.json",
            "--split_index", str(args.split_index),
            "--partner_index", str(partner_index),
        ])
        seed += 1

    return jobs

async def run_job(job, sem):
    async with sem:  
        proc = await asyncio.create_subprocess_shell(" ".join(job))
        await proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split_index", type=int, required=True)
    parser.add_argument("--max_processes", type=int, default=10)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="rl_data")
    parser.add_argument("--num_partners", type=int, default=7)

    args = parser.parse_args()

    asyncio.run(main(args))
