from dataclasses import dataclass
import random

@dataclass
class Drop:
    name: str
    base_p: float
    stock: int
    duration_s: int
    sold: int = 0
    redemptions: int = 0

def urgency_multiplier(t_remaining: int, total: int, k: float = 1.25) -> float:
    if total <= 0:
        return 1.0
    x = max(0.0, min(1.0, t_remaining / total))
    mult = (1.0 + k * (1.0 - x))
    return max(0.2, min(2.0, mult))

def observed_conversion(p_base: float, t_remaining: int, total: int) -> float:
    return max(0.0, min(1.0, p_base * urgency_multiplier(t_remaining, total)))

class ThompsonBeta:
    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
    @classmethod
    def with_k(cls, k: int, a0: float = 1.0, b0: float = 1.0):
        return cls([a0]*k, [b0]*k)
    def sample_index(self, available_idx):
        best_idx, best_val = None, -1
        for i in available_idx:
            val = random.betavariate(self.alpha[i], self.beta[i])
            if val > best_val:
                best_val, best_idx = val, i
        return best_idx
    def update(self, i: int, success: int):
        if success: self.alpha[i] += 1.0
        else: self.beta[i] += 1.0

def random_policy(k: int) -> int:
    return random.randrange(k)

def make_default_drops(k: int = 3, stock: int = 120, duration_s: int = 900):
    base_defaults = [0.06, 0.10, 0.14, 0.08, 0.12, 0.05]
    if k > len(base_defaults):
        base_ps = (base_defaults * ((k + len(base_defaults) - 1)//len(base_defaults)))[:k]
    else:
        base_ps = base_defaults[:k]
    return [Drop(name=f"Drop {i+1}", base_p=base_ps[i], stock=stock, duration_s=duration_s) for i in range(k)]

def simulate_run(drops, users=500, horizon_s=900, policy="random", seed=42):
    random.seed(seed)
    agent = ThompsonBeta.with_k(len(drops)) if policy == "thompson" else None
    arrivals = sorted([random.randrange(horizon_s) for _ in range(users)])
    views = tokens = redemptions = 0
    for t in arrivals:
        available = [i for i,d in enumerate(drops) if d.stock > 0]
        if not available: continue
        if policy == "random":
            i = available[random_policy(len(available))]
        elif policy == "thompson":
            i = agent.sample_index(available)
        d = drops[i]
        views += 1
        t_remaining = max(0, d.duration_s - t)
        p_obs = observed_conversion(d.base_p, t_remaining, d.duration_s)
        token = 1 if random.random() < p_obs else 0
        if token:
            tokens += 1
            d.stock -= 1
            d.sold += 1
            redemptions += 1
            d.redemptions += 1
            if policy == "thompson": agent.update(i, 1)
        else:
            if policy == "thompson": agent.update(i, 0)
    ctr = (tokens / views) if views else 0.0
    conv = (redemptions / tokens) if tokens else 0.0
    util_stock = sum(d.sold for d in drops) / max(1, sum(d.stock + d.sold for d in drops))
    return {"views": views, "tokens": tokens, "redemptions": redemptions,
            "CTR": ctr, "conversion_given_token": conv, "utilization_stock": util_stock}

def compare_policies(users: int, horizon_s: int, drops_k: int, seed: int):
    r_drops = make_default_drops(k=drops_k, duration_s=horizon_s)
    t_drops = make_default_drops(k=drops_k, duration_s=horizon_s)
    return {
        "random": simulate_run(r_drops, users=users, horizon_s=horizon_s, policy="random", seed=seed),
        "thompson": simulate_run(t_drops, users=users, horizon_s=horizon_s, policy="thompson", seed=seed),
    }
