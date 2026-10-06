import time
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from recursive_ssd.io import Deadline, DeadlineReached, digest
from recursive_ssd.methods import Decode, IDS, CONTROLS
from recursive_ssd.model import Policy
from recursive_ssd.train import record_loss, train_round

torch.set_num_threads(1)

def policy():
    torch.manual_seed(17)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        attention_dropout=0.,max_position_embeddings=128))
    return Policy(base,rank=2,device="cpu")


RECORDS=[{"prompt_ids":[1,2,3],"completion_ids":[4,5,6],"loss_weight":1.},
         {"prompt_ids":[3,2],"completion_ids":[7,8,9,10],"loss_weight":1.}]
CONFIG={"learning_rate":.01,"epochs":1,"grad_accum":1,"vocab_chunk":2,
        "max_grad_norm":1.,"method_parameters":{"step_kl_budget":.01}}


def test_causal_completion_loss_matches_full_logits():
    p=policy()
    actual=record_loss(p,RECORDS[0],"hard",Decode(1.5,5,.8),{},2)
    ids=torch.tensor([[1,2,3,4,5]])
    hidden=p.hidden(ids,"student",train=True)
    logits=p.logits(hidden)[0,2:]
    expected=torch.nn.functional.cross_entropy(logits,torch.tensor([4,5,6]))
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize("method",list(IDS.values())+sorted(CONTROLS))
def test_all_methods_backward_and_base_teacher_frozen(method):
    p=policy()
    frozen={n:x.detach().clone() for n,x in p.model.named_parameters() if ".student." not in n}
    loss=record_loss(p,RECORDS[0],method,Decode(1.5,5,.8),{},chunk=2)
    loss.backward()
    assert any(x.grad is not None and x.grad.abs().sum()>0 for n,x in p.model.named_parameters() if ".student." in n)
    for n,x in p.model.named_parameters():
        if n in frozen:
            assert x.grad is None
            torch.testing.assert_close(x,frozen[n])


@pytest.mark.parametrize("method",["hard","M03","M11","M12","M13"])
def test_optimizer_paths_and_recursive_lineage(method,tmp_path):
    p=policy()
    initial={k:v.clone() for k,v in p.state().items()}
    first=train_round(p,RECORDS,method,Decode(1.5,5,.8),CONFIG,tmp_path/"r1",Deadline(time.time()+60),17,"base")
    assert any(not torch.equal(initial[k],v) for k,v in p.state().items())
    p.copy("student","teacher")
    for key,value in p.state().items():
        torch.testing.assert_close(value,p.state("teacher")[key])
    second=train_round(p,RECORDS,method,Decode(1.5,5,.8),CONFIG,tmp_path/"r2",Deadline(time.time()+60),18,digest(first))
    state=torch.load(second,weights_only=True)
    assert state["parent_sha"]==digest(first)
    assert state["successful_updates"]>0


def test_exact_optimizer_resume(tmp_path):
    class StopAfterOne:
        def __init__(self): self.calls=0
        def check(self,*args):
            self.calls+=1
            if self.calls>1: raise DeadlineReached()
    full=policy()
    train_round(full,RECORDS,"hard",Decode(1.5,5,.8),CONFIG,tmp_path/"full",Deadline(time.time()+60),17,"base")
    resumed=policy()
    with pytest.raises(DeadlineReached):
        train_round(resumed,RECORDS,"hard",Decode(1.5,5,.8),CONFIG,tmp_path/"resume",StopAfterOne(),17,"base")
    resumed=policy()
    train_round(resumed,RECORDS,"hard",Decode(1.5,5,.8),CONFIG,tmp_path/"resume",Deadline(time.time()+60),17,"base")
    for key,value in full.state().items():
        torch.testing.assert_close(value,resumed.state()[key],rtol=0,atol=0)


def test_generation_with_cache_and_historical_mixture():
    class Tokenizer:
        eos_token_id=30
        def decode(self,ids,**kwargs): return " ".join(map(str,ids))
    p=policy()
    p.tokenizer=Tokenizer()
    first=p.generate([1,2,3],Decode(1.5,5,.8),5,42,Deadline(time.time()+60),exploration=.1)
    second=p.generate([1,2,3],Decode(1.5,5,.8),5,42,Deadline(time.time()+60),exploration=.1)
    assert first==second
    assert 1<=len(first["completion_ids"])<=5


def test_half_precision_base_with_float_adapter_loss():
    torch.manual_seed(17)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        attention_dropout=0.,max_position_embeddings=128)).half()
    p=Policy(base,rank=2,device="cpu")
    loss=record_loss(p,RECORDS[0],"M03",Decode(1.5,5,.8),{},2)
    assert loss.dtype==torch.float32
    loss.backward()
    assert all(x.dtype==torch.float32 and torch.isfinite(x.grad).all()
               for x in p.trainable() if x.grad is not None)


def test_fixed_prefix_diagnostics_at_initial_policy():
    from recursive_ssd.runner import distribution_diagnostics
    p=policy()
    config={"temperature":1.5,"top_k":5,"top_p":.8,"vocab_chunk":2,
            "method_parameters":{"floor":.1}}
    metrics=distribution_diagnostics(p,RECORDS,config,Deadline(time.time()+60))
    assert metrics["prefixes"]==7
    assert abs(metrics["kl_initial_to_student"])<1e-7
    assert metrics["floor_violation_mass"]==0.
