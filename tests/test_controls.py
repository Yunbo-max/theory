"""CPU engineering checks of controls; none are scientific benchmark evidence."""
import json
import math
import random
import time

import pytest
import torch

from recursive_ssd.io import Deadline, DeadlineReached
from recursive_ssd.methods import Decode, target
from recursive_ssd.train import record_loss, train_round


torch.set_num_threads(1)


class DistributionPolicy:
    """A small trainable categorical model exercising the real optimizer code."""
    def __init__(self):
        self.device = torch.device("cpu")
        self.teacher = torch.tensor([
            [.40, .28, .17, .10, .05], [.12, .52, .20, .11, .05],
            [.20, .12, .48, .13, .07], [.10, .15, .21, .49, .05],
            [.31, .17, .25, .18, .09],
        ]).log()
        self.anchor = self.teacher.flip(-1).clone()
        self.lag = self.teacher.roll(1, -1).clone()
        self.student = torch.nn.Parameter(self.teacher.clone())

    def hidden(self, ids, adapter="student", train=False):
        return getattr(self, adapter)[ids]

    def logits(self, hidden):
        return hidden

    def trainable(self):
        return [self.student]

    def save_checkpoint(self, path, **fields):
        torch.save({"adapter": self.student.detach().clone(), **fields}, path)

    def load_checkpoint(self, path):
        result = torch.load(path, weights_only=True)
        with torch.no_grad():
            self.student.copy_(result["adapter"])
        return result


RECORDS = [
    {"prompt_ids": [0], "completion_ids": [1, 2, 3, 4]},
    {"prompt_ids": [1], "completion_ids": [2, 4, 0]},
]
CONFIG = {"learning_rate": .02, "epochs": 1, "grad_accum": 1,
          "vocab_chunk": 2, "max_grad_norm": 10., "method_parameters": {}}
DECODE = Decode(1.5, 3, 1.)


@pytest.mark.parametrize("method,expected", [
    ("soft_current_mix", [.475, .275, .25]),
    ("head_shape_only", [.375, .125, .5]),
    ("mass_transfer_only", [.3, .45, .25]),
    ("head_identity", [.2, .3, .5]),
    ("arithmetic_same_smoothing", [.36, .4, .24]),
    ("smooth_teacher", [.62, .3, .08]),
    ("geometric_weight_one", [.62, .3, .08]),
    ("ratio_bound_unlimited", [.62, .3, .08]),
    ("temporal_current_twice", [.75, .25, 0.]),
    ("lag_only", [.3, .7, 0.]),
    ("prefix_weight_one", [.75, .25, 0.]),
    ("constant_mean_weight", [.75, .25, 0.]),
    ("arithmetic_anchor_sgd", [.425, .375, .2]),
    ("sgd_norm_matched", [.75, .25, 0.]),
    ("gkd_fkl", [.64, .26, .1]),
    ("gkd_rkl", [.64, .26, .1]),
    ("gkd_jsd", [.64, .26, .1]),
])
def test_control_targets_have_the_declared_information_and_mass(method, expected):
    result = target(method, torch.tensor([[.2, .3, .5]], requires_grad=True),
                    torch.tensor([[.75, .25, 0.]]), torch.tensor([2]),
                    anchor=torch.tensor([[.1, .5, .4]]),
                    lag=torch.tensor([[.3, .7, 0.]]),
                    alpha=.5, floor=.5, epsilon=.2)
    torch.testing.assert_close(result, torch.tensor([expected]))
    assert not result.requires_grad


def test_temperature_head_control_freezes_support_and_scalar_beta():
    result = target("temperature_matched_head", torch.tensor([[.2, .3, .5]]),
                    torch.tensor([[.8, .2, 0.]]), torch.tensor([2]), temperature_beta=.5)
    torch.testing.assert_close(result, torch.tensor([[2/3, 1/3, 0.]]))


def test_iid_tail_keeps_exact_head_and_uses_fresh_draws():
    torch.manual_seed(17)
    mu = torch.tensor([[.3, .25, .2, .15, .1, 0.]]).expand(6000, -1)
    q = target("head_exact_tail_iid", mu, mu, torch.zeros(6000, dtype=torch.long), head=1, draws=4)
    torch.testing.assert_close(q[:, 0], mu[:, 0])
    torch.testing.assert_close(q.mean(0), mu[0], atol=.005, rtol=0)
    assert (q[:, -1] == 0).all()
    hard = target("fresh_hard", mu, mu, torch.full((6000,), 5))
    assert (hard[:, -1] == 0).all()
    torch.testing.assert_close(hard.square().sum(-1), torch.ones(6000))


def test_constant_gate_matches_whole_frozen_sequence_mean():
    p = torch.tensor([[.2, .3, .5], [.1, .5, .4]])
    mu = torch.tensor([[.8, .2, 0.], [.3, .7, 0.]])
    lag = torch.tensor([[.2, .8, 0.], [.3, .7, 0.]])
    # JS((.8,.2),(.2,.8))=.192744757; second position has JS=0.
    expected_a = (1 / (1 + math.exp(.192744757 / .1)) + .5) / 2
    q = target("constant_mean_gate", p, mu, torch.tensor([0, 1]), lag=lag)
    torch.testing.assert_close(q, (1-expected_a)*p + expected_a*mu)


def test_budget_matched_fresh_alpha_is_constant_across_entire_sequence():
    p = torch.tensor([[.2, .3, .5], [.1, .5, .4]])
    mu = torch.tensor([[.8, .2, 0.], [.3, .7, 0.]])
    # Sum G(mu)=.32+.42=.74; budget=.02*2=.04. Last token is never drawn.
    q = target("fresh_alpha_budget_matched", p, mu, torch.tensor([2, 2]), noise_budget=.02)
    a = math.sqrt(.04 / .74)
    torch.testing.assert_close(q[:, 2], (1-a)*p[:, 2])


@pytest.mark.parametrize("method", ["noise_allocation", "fresh_alpha_budget_matched",
                                   "constant_mean_gate", "constant_mean_weight",
                                   "head_exact_tail_iid", "tail_stratified", "fresh_hard"])
def test_sequence_coefficients_and_fresh_labels_do_not_depend_on_chunk_size(method):
    losses, grads = [], []
    for chunk in (1, 3, 8):
        p = DistributionPolicy()
        torch.manual_seed(902)
        loss = record_loss(p, RECORDS[0], method, DECODE, {"head": 1, "draws": 4}, chunk)
        loss.backward()
        losses.append(loss.detach())
        grads.append(p.student.grad)
    for loss, grad in zip(losses[1:], grads[1:]):
        torch.testing.assert_close(loss, losses[0], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(grad, grads[0], atol=1e-7, rtol=1e-6)


def test_constant_prefix_weight_matches_exclusive_sequence_mean():
    p = DistributionPolicy()
    actual = record_loss(p, RECORDS[0], "constant_mean_weight", DECODE, {}, 2)
    # Teacher/anchor ratios at observed labels are .28/.10, .20/.20,
    # .13/.12 and .05/.10; exclusive weights are 1, 1/2.8, 1/2.8, 1/(2.8*13/12).
    expected_weight = (1 + 1/2.8 + 1/2.8 + 1/(2.8*13/12))/4
    full = record_loss(p, RECORDS[0], "full_soft", DECODE, {}, 2)
    torch.testing.assert_close(actual, full*expected_weight)


@pytest.mark.parametrize("method", ["gkd_fkl", "gkd_rkl", "gkd_jsd"])
def test_gkd_divergence_matches_manual_objective_and_detaches_round_teacher(method):
    p = DistributionPolicy()
    p.teacher.requires_grad_()
    with torch.no_grad():
        p.student.add_(torch.tensor([.2, -.1, .05, -.2, 0.]))
    value = record_loss(p, RECORDS[0], method, Decode(1., 0, 1.), {"epsilon": .02}, 2)
    log_s = p.student[:4].log_softmax(-1)
    log_q = p.teacher.detach()[:4].log_softmax(-1)
    s, q = log_s.exp(), log_q.exp()
    if method == "gkd_fkl":
        expected = (q*(log_q-log_s)).sum(-1).mean()
    elif method == "gkd_rkl":
        expected = (s*(log_s-log_q)).sum(-1).mean()
    else:
        log_mid = ((s+q)/2).log()
        expected = ((q*(log_q-log_mid)).sum(-1) + (s*(log_s-log_mid)).sum(-1)).mean()/2
    torch.testing.assert_close(value, expected, atol=2e-7, rtol=1e-5)
    expected_grad, = torch.autograd.grad(expected, p.student)
    value.backward()
    torch.testing.assert_close(p.student.grad, expected_grad, atol=2e-7, rtol=1e-5)
    assert p.teacher.grad is None


@pytest.mark.parametrize("method", ["head_identity", "gkd_fkl", "gkd_rkl", "gkd_jsd"])
def test_untransformed_identical_teacher_has_no_learning_signal(method):
    p = DistributionPolicy()
    loss = record_loss(p, RECORDS[0], method, Decode(1., 0, 1.), {}, 2)
    loss.backward()
    assert p.student.grad.abs().max() < 1e-7


def test_reverse_kl_rejects_unsmoothed_truncation():
    with pytest.raises(ValueError, match="epsilon"):
        record_loss(DistributionPolicy(), RECORDS[0], "gkd_rkl", DECODE, {"epsilon": 0.}, 2)


def test_sgd_norm_match_preserves_direction_and_charges_projection_gradient(tmp_path):
    # A current student moved away from the initial policy gives a nonzero
    # retention gradient; choose anchor to oppose the hard/soft target step.
    raw = DistributionPolicy()
    matched = DistributionPolicy()
    projected = DistributionPolicy()
    for p in (raw, matched, projected):
        p.anchor = p.teacher.flip(-1).clone()
    before = raw.student.detach().clone()
    cfg = {**CONFIG, "optimizer": "sgd"}
    for label, p in (("full_soft_sgd", raw), ("sgd_norm_matched", matched),
                     ("gradient_projection", projected)):
        train_round(p, RECORDS[:1], label, DECODE, cfg, tmp_path/label,
                    Deadline(time.time()+90), 17, "base")
    d_raw, d_match, d_proj = (p.student.detach()-before for p in (raw, matched, projected))
    assert d_match.norm() < d_raw.norm()
    torch.testing.assert_close(d_match.norm(), d_proj.norm(), rtol=5e-5, atol=1e-7)
    # Inspect the actual optimizer direction before subtracting nearby float32
    # parameters; normalization would amplify a parameter ULP by ~1/step_size.
    g_match, g_raw = matched.student.grad, raw.student.grad
    torch.testing.assert_close(g_match/g_match.norm(), g_raw/g_raw.norm(), rtol=2e-6, atol=1e-7)
    summary = json.loads((tmp_path/"sgd_norm_matched"/"training.json").read_text())
    assert summary["costs"]["retention_backward_passes"] == 1


def test_learning_rate_multiplier_applies_to_the_requested_optimizer(tmp_path):
    before = DistributionPolicy().student.detach()
    changes = []
    for multiplier in (1., .5):
        p = DistributionPolicy()
        cfg = {**CONFIG, "optimizer": "sgd", "learning_rate_multiplier": multiplier}
        train_round(p, RECORDS[:1], "full_soft", DECODE, cfg, tmp_path/str(multiplier),
                    Deadline(time.time()+90), 17, "base")
        changes.append(p.student.detach()-before)
    # Each stored parameter was rounded to float32. In this range, subtracting
    # the half step from half the full step has at most .75 ULP rounding error;
    # use a one-ULP envelope rather than an arbitrary absolute tolerance.
    ulp=(torch.nextafter(before,torch.full_like(before,torch.inf))-before).abs()
    assert ((changes[1]-changes[0]/2).abs() <= ulp).all()


def test_identity_arm_never_counts_roundoff_as_training_gain(tmp_path):
    p = DistributionPolicy()
    before = p.student.detach().clone()
    train_round(p, RECORDS, "head_identity", DECODE, CONFIG, tmp_path,
                Deadline(time.time()+90), 17, "base")
    torch.testing.assert_close(p.student, before, rtol=0, atol=0)
    summary = json.loads((tmp_path/"training.json").read_text())
    assert summary["successful_updates"] == 0
    assert summary["zero_gradient_updates"] == 2
    assert summary["training_outcome"] == "identity_zero_update"


def test_per_update_rollout_resume_never_regenerates_completed_updates(tmp_path):
    class StopAfterOne:
        calls = 0
        def check(self, *args):
            self.calls += 1
            if self.calls > 1:
                raise DeadlineReached()

    calls = []
    def rollout(group, batch):
        calls.append(group)
        # Exercise checkpointed Python and torch RNG, with actual fresh labels.
        token = 1 + int(torch.randint(4, (1,)))
        return [{**row, "completion_ids": [token, random.randint(1, 4)]} for row in batch]

    def initialize_rng():
        torch.manual_seed(61)
        random.seed(71)

    initialize_rng()
    full = DistributionPolicy()
    train_round(full, RECORDS, "gkd_fkl", DECODE, CONFIG, tmp_path/"full",
                Deadline(time.time()+90), 17, "base", rollout_callback=rollout)
    assert calls == [0, 1]
    calls.clear()
    initialize_rng()
    resumed = DistributionPolicy()
    with pytest.raises(DeadlineReached):
        train_round(resumed, RECORDS, "gkd_fkl", DECODE, CONFIG, tmp_path/"resume",
                    StopAfterOne(), 17, "base", rollout_callback=rollout)
    assert calls == [0]
    torch.manual_seed(8)
    random.seed(8)
    resumed = DistributionPolicy()
    train_round(resumed, RECORDS, "gkd_fkl", DECODE, CONFIG, tmp_path/"resume",
                Deadline(time.time()+90), 17, "base", rollout_callback=rollout)
    assert calls == [0, 1]
    torch.testing.assert_close(resumed.student, full.student, rtol=0, atol=0)
    train_round(resumed, RECORDS, "gkd_fkl", DECODE, CONFIG, tmp_path/"resume",
                Deadline(time.time()+90), 17, "base", rollout_callback=rollout)
    assert calls == [0, 1]


def test_training_clock_is_finite_and_stops_between_updates(tmp_path):
    with pytest.raises(ValueError, match="training_wall_seconds"):
        train_round(DistributionPolicy(), RECORDS, "hard", DECODE,
                    {**CONFIG, "training_wall_seconds": math.inf}, tmp_path/"bad",
                    Deadline(time.time()+90), 17, "base")
    cfg = {**CONFIG, "training_wall_seconds": .00001, "max_epochs": 10}
    p = DistributionPolicy()
    # The first admitted update may cross the cap; no second update is admitted.
    train_round(p, RECORDS, "hard", DECODE, cfg, tmp_path/"clock",
                Deadline(time.time()+90), 17, "base")
    summary = json.loads((tmp_path/"clock"/"training.json").read_text())
    assert summary["attempted_updates"] == 1
    assert summary["stop_reason"] == "training_wall_seconds"
    assert summary["elapsed_seconds"] >= cfg["training_wall_seconds"]


def test_resume_rejects_changed_optimizer_or_control_configuration(tmp_path):
    p = DistributionPolicy()
    train_round(p, RECORDS, "full_soft", DECODE, CONFIG, tmp_path,
                Deadline(time.time()+90), 17, "base")
    with pytest.raises(ValueError, match="configuration"):
        train_round(DistributionPolicy(), RECORDS, "full_soft", DECODE,
                    {**CONFIG, "optimizer": "sgd"}, tmp_path,
                    Deadline(time.time()+90), 17, "base")
