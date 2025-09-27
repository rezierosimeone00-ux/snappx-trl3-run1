
"""
Snappx — TRL‑3 / Run 1
Random vs Thompson (Beta-Bernoulli) on drop feed with urgency (countdown) and scarcity (stock).
This module contains the simulation primitives shared by the Colab notebook and the Streamlit app.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import random

@dataclass
class Drop:
    """A single drop/offer with an intrinsic base conversion rate p (unknown to the agent)."""
    name: str
    base_p: float  # intrinsic conversion probability (unknown to agent)
    stock: int     # number of tokens available (scarcity)
    duration_s: int  # total duration in seconds (urgency)
    # derived/updated during simulation
    sold: int = 0
    redemptions: int = 0

def urgency_multiplier(t_remaining: int, total: int, k: float = 1.25) -> float:
    """
    Urgency effect: increases conversion as time runs out.
    Simple monotonic function in (0, 1] → [0.6, ~1.3] depending on k.
    We clamp to [0.2, 2.0] to avoid extremes.
    """
    if total <= 0:
        return 1.0
    x = max(0.0, min(1.0, t_remaining / total))
    # Sharper near the end; k controls curvature
    mult = (1.0 + k * (1.0 - x))
    return max(0.2, min(2.0, mult))

def observed_conversion(p_base: float, t_remaining: int, total: int) -> float:
    """Apply urgency multiplier to the base probability."""
    return max(0.0, min(1.0, p_base * urgency_multiplier(t_remaining, total)))

@dataclass
class ThompsonBeta:
    """Independent Beta-Bernoulli Thompson Sampling agent for K drops."""
    alpha: List[float]
    beta: List[float]

    @classmethod
    def with_k(cls, k: int, a0: float = 1.0, b0: float = 1.0):
        return cls(alpha=[a0]*k, beta=[b0]*k)

    def select(self) -> int:
        samples = [random.betavariate(a, b) for a, b in zip(self.alpha, self.beta)]
        return max(range(len(samples)), key=lambda i: samples[i])

    def update(self, i: int, success: int):
        if success:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0

def random_policy(k: int) -> int:
    return random.randrange(k)

def simulate_run(
    drops: List[Drop],
    users: int = 500,
    horizon_s: int = 900,
    policy: str = "random",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Simulate a single run where each "user" arrives uniformly over the time horizon.
    Each user sees all drops that still have stock and selects one shown by the policy.
    Success is a Bernoulli trial with urgency-adjusted probability.
    Returns KPIs: views, clicks (tokens), redemptions, CTR, conv, etc.
    """
    random.seed(seed)

    # Agent for Thompson
    agent = ThompsonBeta.with_k(len(drops)) if policy == "thompson" else None

    # Arrival times for users
    arrivals = sorted([random.randrange(horizon_s) for _ in range(users)])

    views = 0
    tokens = 0
    redemptions = 0

    for t in arrivals:
        # Collect available drops (stock>0), compute urgency‑adjusted p
        available = [(i, d) for i, d in enumerate(drops) if d.stock > 0]
        if not available:
            continue

        # Policy selects an index
        if policy == "random":
            choice_idx = random_policy(len(available))
        elif policy == "thompson":
            # Ask agent for choice over full set; remap if some are out of stock
            # We sample over all indices but only consider available; to keep it simple:
            # rebuild a shadow agent distribution for available indices.
            # (For a proper approach, maintain per-index state; for TRL‑3 this is sufficient.)
            mapped = [i for i, _ in available]
            samples = []
            for i in mapped:
                a = agent.alpha[i]
                b = agent.beta[i]
                samples.append(random.betavariate(a, b))
            choice_idx = max(range(len(samples)), key=lambda j: samples[j])
        else:
            raise ValueError("Unknown policy")

        i, d = available[choice_idx]
        views += 1

        t_remaining = max(0, d.duration_s - t)
        p_obs = observed_conversion(d.base_p, t_remaining, d.duration_s)

        # User decides to buy token (click) with probability p_obs
        token = 1 if random.random() < p_obs else 0
        if token:
            tokens += 1
            d.stock -= 1
            d.sold += 1

            # For TRL‑3 Run 1, we align "token" with "redemption" in one step
            # (we will separate them in later runs).
            redemption = 1
            redemptions += 1
            d.redemptions += 1

            if policy == "thompson":
                agent.update(i, success=1)
        else:
            if policy == "thompson":
                agent.update(i, success=0)

    ctr = (tokens / views) if views else 0.0
    conv = (redemptions / max(1, tokens)) if tokens else 0.0
    util_stock = sum(d.sold for d in drops) / max(1, sum(d.stock + d.sold for d in drops))

    return {
        "views": views,
        "tokens": tokens,
        "redemptions": redemptions,
        "CTR": ctr,
        "conversion_given_token": conv,
        "utilization_stock": util_stock,
    }

def make_default_drops(k: int = 3) -> List[Drop]:
    """
    Build a small set of drops with different base probabilities and identical durations/stock.
    """
    base_ps = [0.06, 0.10, 0.14][:k]
    total = 900  # 15 minutes
    stock = 120
    return [Drop(name=f"Drop {i+1}", base_p=base_ps[i], stock=stock, duration_s=total) for i in range(k)]

def compare_policies(
    users: int = 500,
    horizon_s: int = 900,
    seed: int = 42,
    k: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Compare Random vs Thompson on the same scenario.
    """
    # Random
    drops_r = make_default_drops(k)
    res_random = simulate_run(drops_r, users=users, horizon_s=horizon_s, policy="random", seed=seed)

    # Thompson
    drops_t = make_default_drops(k)
    res_thomp = simulate_run(drops_t, users=users, horizon_s=horizon_s, policy="thompson", seed=seed)

    return {"random": res_random, "thompson": res_thomp}
