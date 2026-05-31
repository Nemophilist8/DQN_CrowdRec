"""Quick stats: requester pool size & worker active-project counts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.platform_env import PlatformEnvConfig, PlatformSimulationEnv
from models.platform_baselines import select_requester_worker, select_worker_project
from scripts.evaluate_platform import build_with_limit
from src.platform_dataset import PlatformDataset


def run(args) -> None:
    cfg = PlatformEnvConfig(
        max_steps_per_episode=None if args.max_steps == 0 else args.max_steps,
        requester_immediate_decision=args.immediate_requester,
        requester_batch_size=args.requester_batch_size,
    )
    ds = build_with_limit(args.max_projects)
    platform = PlatformDataset(ds, args.split)

    # Requester pool under WAIT
    env = PlatformSimulationEnv(platform, cfg, seed=args.seed)
    d = env.reset()
    req_pool: list[int] = []
    waits: list[bool] = []
    steps = 0
    limit = None if args.max_steps == 0 else args.max_steps
    while d is not None and (limit is None or steps < limit):
        if d.actor == "requester":
            pool = [w for w in d.info.get("candidate_worker_ids", []) if w is not None]
            req_pool.append(len(pool))
            a = select_requester_worker("wait_until_deadline", env, d)
            waits.append(a == 0)
        else:
            a = select_worker_project("industry_match", env, d)
        d = env.step(a).decision
        steps += 1

    rp = np.array(req_pool) if req_pool else np.array([0.0])
    print(f"=== WAIT requester pool [{args.split}, n={args.max_projects or 'all'}, steps={steps}] ===")
    print(f"  decisions={len(rp)} mean={rp.mean():.2f} median={np.median(rp):.0f} max={rp.max():.0f}")
    print(f"  pool>=2={(rp >= 2).mean() * 100:.1f}%  pool>=5={(rp >= 5).mean() * 100:.1f}%  pool>=8={(rp >= 8).mean() * 100:.1f}%")
    if waits:
        print(f"  WAIT rate={np.mean(waits) * 100:.1f}%")

    # Worker valid candidate count
    platform2 = PlatformDataset(build_with_limit(args.max_projects), args.split)
    env2 = PlatformSimulationEnv(platform2, cfg, seed=args.seed)
    d = env2.reset()
    hist: Counter[int] = Counter()
    steps = 0
    while d is not None and (limit is None or steps < limit):
        if d.actor == "worker":
            hist[int(d.observation.action_mask.sum())] += 1
            valid = np.flatnonzero(d.observation.action_mask)
            a = int(valid[0]) if len(valid) else 0
        else:
            a = 0
        d = env2.step(a).decision
        steps += 1

    total = sum(hist.values())
    print(f"=== Worker active projects [{args.split}] (n={total} decisions) ===")
    for k, c in sorted(hist.items())[:8]:
        print(f"  {k} candidates: {c} ({c / total * 100:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--max-projects", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--requester-batch-size", type=int, default=8)
    p.add_argument("--immediate-requester", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())


if __name__ == "__main__":
    main()
