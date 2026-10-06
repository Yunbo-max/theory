"""Engineering/math checks. Synthetic finite distributions are NOT benchmarks."""
import itertools
import math
import pytest
import torch
from recursive_ssd.methods import *

torch.set_num_threads(1)


@pytest.mark.parametrize("method",list(IDS.values())+sorted(CONTROLS))
def test_targets_normalized_detached(method):
    torch.manual_seed(4)
    logits=torch.randn(7,11,requires_grad=True)
    p=logits.softmax(-1)
    mu=decoder(logits,Decode(1.5,5,.8))
    anchor=torch.randn(7,11).softmax(-1)
    lag=decoder(torch.randn(7,11),Decode(1.5,5,.8))
    q=target(method,p,mu,mu.argmax(-1),anchor=anchor,lag=lag)
    assert not q.requires_grad
    assert torch.isfinite(q).all() and (q>=0).all()
    torch.testing.assert_close(q.sum(-1),torch.ones(7),atol=2e-6,rtol=0)


def test_decoder_order_and_crossing_token():
    p=torch.tensor([[.5,.3,.2]])
    q=decoder(p.log(),Decode(1.,2,.7))
    torch.testing.assert_close(q,torch.tensor([[.625,.375,0.]]))
    q=decoder(p.log(),Decode(1.,0,.49))
    torch.testing.assert_close(q,torch.tensor([[1.,0.,0.]]))


def test_floor_kkt_and_small_simplex_search():
    mu=torch.tensor([[.85,.15,0.]],dtype=torch.float64)
    anchor=torch.tensor([[.1,.3,.6]],dtype=torch.float64)
    q=floor_projection(mu,anchor,.3)
    assert (q>=.3*anchor-1e-10).all()
    torch.testing.assert_close(q.sum(-1),torch.ones(1,dtype=torch.float64),atol=1e-8,rtol=0)
    free=q>.3*anchor+1e-8
    ratios=(q/mu.clamp_min(1e-20))[free]
    torch.testing.assert_close(ratios,ratios.mean().expand_as(ratios),atol=1e-9,rtol=0)
    best=float(kl(mu,q))
    for a in range(4,83):
        for b in range(9,101-a):
            candidate=torch.tensor([[a,b,100-a-b]],dtype=torch.float64)/100
            if (candidate>=.3*anchor).all():
                assert best<=float(kl(mu,candidate))+1e-7


def test_ratio_bound_and_conditional_diversity():
    p=torch.tensor([[.85,.1,.04,.01]])
    mu=torch.tensor([[.9,.1,0.,0.]])
    anchor=torch.tensor([[.3,.4,.2,.1]])
    q=target("M02",p,mu,torch.tensor([0]),anchor=anchor,ratio_bound=.7)
    assert (q/anchor<=math.exp(1.4)+1e-6).all()
    assert (q/anchor>=math.exp(-1.4)-1e-6).all()
    q=diversity_floor(mu,anchor,.9)
    conditional=anchor[:,:2]/anchor[:,:2].sum(-1,keepdim=True)
    assert gini(q)>=.9*gini(conditional)-1e-6
    assert (q[:,2:]==0).all()


def test_sampling_expectation_and_local_noise_identity():
    p=torch.tensor([.2,.5,.3],dtype=torch.float64)
    mu=torch.tensor([.6,.4,0.],dtype=torch.float64)
    a=.4
    outcomes=(1-a)*p[None,:]+a*torch.eye(3,dtype=torch.float64)
    mean=(outcomes*mu[:,None]).sum(0)
    torch.testing.assert_close(mean,(1-a)*p+a*mu)
    var=((outcomes-mean).square().sum(-1)*mu).sum()
    torch.testing.assert_close(var,a*a*gini(mu))
    torch.testing.assert_close((gini(outcomes)*mu).sum(),gini(mean)-var)


def test_stratified_unbiased_and_exact_head():
    torch.manual_seed(5)
    mu=torch.tensor([[.3,.25,.2,.15,.1]]).expand(10000,-1)
    q=stratified_target(mu,head=1,draws=4)
    torch.testing.assert_close(q[:,0],mu[:,0])
    torch.testing.assert_close(q.mean(0),mu[0],atol=.003,rtol=0)


def test_noise_allocation_feasibility():
    p=torch.tensor([[.2,.3,.5],[.1,.7,.2],[.5,.25,.25]])
    mu=torch.tensor([[.8,.2,0.],[.3,.7,0.],[.9,.1,0.]])
    a=allocate_noise(p,mu,.01)
    assert (gini(mu)*a.square()).sum()<=.03+1e-7
    assert (a>=0).all() and (a<=1).all()


def test_softmax_gradient_chain_rule():
    logits=torch.tensor([[-2.,1.,.1]],requires_grad=True)
    q=torch.tensor([[.6,.25,.15]])
    loss=-(q*logits.log_softmax(-1)).sum()
    loss.backward()
    torch.testing.assert_close(logits.grad,logits.detach().softmax(-1)-q)
    assert logits.grad.abs().max()<=1


def test_prefix_exclusive_and_parameter_halfspace():
    logs=torch.tensor([1.,2.,-3.])
    torch.testing.assert_close(prefix_weights(logs),torch.tensor([1.,math.exp(-1),math.exp(-3)]))
    d=[torch.tensor([2.,-1.])]
    h=[torch.tensor([1.,1.])]
    projected=project_direction(d,h)
    assert (projected[0]*h[0]).sum()<=1e-6
    torch.testing.assert_close(projected[0],torch.tensor([1.5,-1.5]))


def test_prompt_counts_and_equal_prompt_weights():
    counts=allocate_prompts([.1,.8,.1],20)
    assert sum(counts)==20 and min(counts)>=1
    assert counts[1]>counts[0]
    for n in counts:
        assert math.isclose(n*(20/(3*n))/20,1/3)


def test_temperature_composition():
    p=torch.tensor([.2,.5,.3],dtype=torch.float64)
    config=round_decode("M04",Decode(1.5,0,1.),3)
    x=p
    for _ in range(3):
        # Use double-precision analytic map for the mathematical identity.
        x=(x.log()/config.temperature).softmax(-1)
    torch.testing.assert_close(x,(p.log()/1.5).softmax(-1))
