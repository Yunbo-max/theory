"""Explicit target-space constructions. All references and targets are detached.

These functions implement conditional distribution-level math, not guarantees on
neural fitting, full-trajectory gradient variance or answer correctness.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


IDS = {
    "M01":"geometric_anchor", "M02":"ratio_cap", "M03":"floor_projection",
    "M04":"temperature_budget", "M05":"head_mass", "M06":"head_diversity",
    "M07":"tail_stratified", "M08":"temporal_mean", "M09":"disagreement_gate",
    "M10":"prefix_damping", "M11":"noise_allocation", "M12":"gradient_projection",
    "M13":"kl_backtrack", "M14":"prompt_allocation", "M15":"ancestral_exploration",
}
CONTROLS = {"hard", "full_soft", "fixed_alpha", "fresh_alpha", "arithmetic_anchor",
            "lower_lr", "fixed_data", "full_soft_sgd"}
ANCHOR = {"geometric_anchor", "ratio_cap", "floor_projection", "head_diversity",
          "prefix_damping", "gradient_projection", "arithmetic_anchor"}
LAG = {"temporal_mean", "disagreement_gate"}
HARD = {"hard", "lower_lr", "fixed_data", "prompt_allocation", "ancestral_exploration"}


@dataclass(frozen=True)
class Decode:
    temperature: float = 1.5
    top_k: int = 20
    top_p: float = 0.8

    def __post_init__(self):
        if not (self.temperature > 0 and math.isfinite(self.temperature)):
            raise ValueError("temperature must be finite and positive")
        if self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid top-k/top-p")


def name(method):
    method = IDS.get(method, method)
    if method not in set(IDS.values()) | CONTROLS:
        raise ValueError(f"unknown method: {method}")
    return method


def round_decode(method, config, rounds):
    if name(method) == "temperature_budget":
        return Decode(config.temperature ** (1 / rounds), config.top_k, config.top_p)
    return config


def decoder(logits, config):
    """Temperature -> top-k -> top-p, keep crossing token and at least one token.

    This is the sampling order of upstream SSD's vLLM configuration. Tied logits
    at a top-k threshold are retained; backend floating-point draws need not be
    bit-identical to vLLM. Training and generation use this same function.
    """
    scores = logits.float() / config.temperature
    if config.top_k and config.top_k < scores.shape[-1]:
        cutoff = scores.topk(config.top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if config.top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(-1, indices, remove)
        scores = scores.masked_fill(mask, -torch.inf)
    return scores.softmax(-1)


def gini(p):
    return (1 - p.square().sum(-1)).clamp_min(0)


def kl(p, q):
    return (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1)


def floor_projection(mu, anchor, floor=0.1, iterations=32):
    if not 0 <= floor < 1:
        raise ValueError("floor must be in [0,1)")
    lo = torch.zeros_like(mu[..., :1])
    hi = torch.ones_like(lo)
    base = floor * anchor
    for _ in range(iterations):
        mid = (lo + hi) / 2
        too_much = torch.maximum(base, mid * mu).sum(-1, keepdim=True) > 1
        hi = torch.where(too_much, mid, hi)
        lo = torch.where(too_much, lo, mid)
    # Tiny FP32 normalization error remains; do not renormalize below the floor.
    return torch.maximum(base, ((lo + hi) / 2) * mu)


def diversity_floor(mu, anchor, retention=0.9):
    support = mu > 0
    ap = anchor * support
    ap = ap / ap.sum(-1, keepdim=True).clamp_min(1e-30)
    goal = retention * gini(ap)
    logs = mu.clamp_min(1e-30).log()
    lo, hi = torch.zeros_like(mu[..., :1]), torch.ones_like(mu[..., :1])
    def at(beta):
        return (logs * beta).masked_fill(~support, -torch.inf).softmax(-1)
    for _ in range(28):
        mid = (lo + hi) / 2
        feasible = gini(at(mid)).unsqueeze(-1) >= goal.unsqueeze(-1)
        lo = torch.where(feasible, mid, lo)
        hi = torch.where(feasible, hi, mid)
    return at(lo)


def stratified_target(mu, head=4, draws=4):
    head = min(head, mu.shape[-1])
    values, indices = mu.topk(head, dim=-1)
    q = torch.zeros_like(mu).scatter(-1, indices, values)
    tail = (mu-q).clamp_min(0)
    mass = tail.sum(-1, keepdim=True)
    conditional = tail / mass.clamp_min(1e-30)
    # For zero-tail rows searchsorted needs a valid CDF, though additions are 0.
    conditional = torch.where(mass > 1e-12, conditional,
                              torch.ones_like(conditional)/conditional.shape[-1])
    cdf = conditional.cumsum(-1).contiguous()
    cdf[..., -1] = 1
    shape = (*mu.shape[:-1], draws)
    u = (torch.arange(draws, device=mu.device) + torch.rand(shape, device=mu.device)) / draws
    sampled = torch.searchsorted(cdf, u.contiguous()).clamp_max(mu.shape[-1]-1)
    return q.scatter_add(-1, sampled, mass.expand_as(u)/draws)


def allocate_noise(p, mu, per_token_budget=0.02):
    d = (mu-p).square().sum(-1)
    v = gini(mu)
    budget = per_token_budget * v.numel()
    if budget < 0:
        raise ValueError("noise budget must be nonnegative")
    if v.sum() <= budget:
        return torch.ones_like(v)
    def at(lam):
        return torch.where(v > 1e-12, (d/(2*lam*v.clamp_min(1e-12))).clamp(0,1), torch.ones_like(v))
    lo, hi = 0.0, 1.0
    while (v*at(hi).square()).sum() > budget and hi < 1e16:
        hi *= 2
    for _ in range(40):
        mid=(lo+hi)/2
        if (v*at(mid).square()).sum() > budget:
            lo=mid
        else:
            hi=mid
    return at(hi)


@torch.no_grad()
def target(method, p, mu, labels, anchor=None, lag=None, *, alpha=0.5, floor=0.1,
           epsilon=0.02, ratio_bound=3.0, head_transfer=0.5,
           diversity_retention=0.9, gate_lambda=0.1, noise_budget=0.02, **_):
    method=name(method)
    if method in ANCHOR and anchor is None:
        raise ValueError(f"{method} requires same-prefix initial distribution")
    if method in LAG and lag is None:
        raise ValueError(f"{method} requires same-prefix lag distribution")
    if method in HARD:
        return F.one_hot(labels, p.shape[-1]).to(p)
    if method in {"full_soft","temperature_budget","prefix_damping","gradient_projection","kl_backtrack","full_soft_sgd"}:
        return mu.detach()
    if method in {"fixed_alpha", "fresh_alpha"}:
        if method=="fresh_alpha":
            labels=torch.multinomial(mu, 1).squeeze(-1)
        return (1-alpha)*p+alpha*F.one_hot(labels,p.shape[-1]).to(p)
    if method=="arithmetic_anchor":
        return (1-floor)*mu+floor*anchor
    if method=="floor_projection":
        return floor_projection(mu, anchor, floor)
    if method=="geometric_anchor":
        m=(1-epsilon)*mu+epsilon*anchor
        return (alpha*m.clamp_min(1e-30).log()+(1-alpha)*anchor.clamp_min(1e-30).log()).softmax(-1)
    if method=="ratio_cap":
        m=(1-epsilon)*mu+epsilon*anchor
        loga=anchor.clamp_min(1e-30).log()
        return (loga+(m.clamp_min(1e-30).log()-loga).clamp(-ratio_bound,ratio_bound)).softmax(-1)
    if method=="head_mass":
        support=mu>0
        mass=(p*support).sum(-1,keepdim=True)
        return (mass+head_transfer*(1-mass))*mu+(1-head_transfer)*p*(~support)
    if method=="head_diversity":
        return diversity_floor(mu,anchor,diversity_retention)
    if method=="tail_stratified":
        return stratified_target(mu)
    if method=="temporal_mean":
        return (mu+lag)/2
    if method=="disagreement_gate":
        mid=(mu+lag)/2
        js=(kl(mu,mid)+kl(lag,mid))/2
        a=torch.sigmoid(torch.as_tensor(math.log(alpha/(1-alpha)),device=p.device)-js/gate_lambda)
        return (1-a[:,None])*p+a[:,None]*mu
    if method=="noise_allocation":
        a=allocate_noise(p,mu,noise_budget)
        # Fresh draws AFTER all prefixes and coefficients have been fixed.
        fresh=torch.multinomial(mu,1).squeeze(-1)
        return (1-a[:,None])*p+a[:,None]*F.one_hot(fresh,p.shape[-1]).to(p)
    raise AssertionError(method)


def prefix_weights(log_ratios, cap=5.0):
    positive=log_ratios.detach().clamp_min(0)
    exclusive=positive.cumsum(0)-positive
    return (-exclusive.clamp_max(cap)).exp()


def project_direction(direction, retention_gradient, allowance=0.0):
    norm=sum(h.float().square().sum() for h in retention_gradient)
    dot=sum((d.float()*h.float()).sum() for d,h in zip(direction,retention_gradient))
    coefficient=((dot-allowance).clamp_min(0)/norm.clamp_min(1e-30)) if norm > 0 else 0.0
    return [d-coefficient*h for d,h in zip(direction,retention_gradient)]


def allocate_prompts(sigmas, total):
    """Integer proxy allocation; not claimed to solve the relaxed optimum exactly."""
    x=torch.as_tensor(sigmas,dtype=torch.float64,device="cpu")
    if total<len(x) or len(x)==0 or (x<0).any() or not x.isfinite().all():
        raise ValueError("invalid allocation")
    shares=x/x.sum() if x.sum()>0 else torch.ones_like(x)/len(x)
    extra=shares*(total-len(x))
    counts=extra.floor().long()+1
    left=total-int(counts.sum())
    indices=torch.argsort(extra-extra.floor(),descending=True,stable=True)[:left]
    counts[indices]+=1
    return counts.tolist()
