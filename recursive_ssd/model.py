"""One frozen base and small adapters; no full teacher model duplication."""
from contextlib import contextmanager
from pathlib import Path
import os
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import MODEL, MODEL_REV
from .io import Deadline, digest
from .methods import Decode, decoder


class Policy:
    def __init__(self, base, tokenizer=None, rank=16, device="cuda"):
        self.tokenizer=tokenizer
        self.device=torch.device(device)
        config=LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=0.0,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM",bias="none")
        self.model=get_peft_model(base,config,adapter_name="student")
        for adapter in ("teacher","lag"):
            self.model.add_adapter(adapter,config)
        self.model.to(self.device)
        self.model.config.use_cache=False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.model.enable_input_require_grads()
        self.initial={k:v.detach().cpu().clone() for k,v in self.state("student").items()}
        self.copy("student","teacher")
        self.copy("student","lag")
        self.select("student",train=True)

    @classmethod
    def load(cls, rank=16):
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV,trust_remote_code=False,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_REV,torch_dtype=torch.float16,
            attn_implementation="sdpa",trust_remote_code=False,local_files_only=True)
        return cls(base,tokenizer,rank=rank)

    def state(self, adapter="student"):
        return get_peft_model_state_dict(self.model,adapter_name=adapter)

    def set_state(self, state, adapter="student"):
        set_peft_model_state_dict(self.model,state,adapter_name=adapter)

    def copy(self, source, destination):
        self.set_state({k:v.detach().clone() for k,v in self.state(source).items()},destination)

    def reset(self):
        for adapter in ("student","teacher","lag"):
            self.set_state(self.initial,adapter)

    def select(self, adapter, train=False):
        self.model.set_adapter(adapter)
        for n,p in self.model.named_parameters():
            p.requires_grad_(train and ".student." in n)
        self.model.train(train)

    @contextmanager
    def using(self, adapter, train=False):
        if adapter=="anchor":
            self.model.eval()
            with self.model.disable_adapter():
                yield
        else:
            self.select(adapter,train)
            yield

    def hidden(self, ids, adapter="student", train=False):
        with self.using(adapter,train):
            return self.model.get_base_model().model(input_ids=ids,use_cache=False).last_hidden_state

    def logits(self, hidden):
        return self.model.get_base_model().lm_head(hidden)

    def trainable(self):
        self.select("student",True)
        return [p for p in self.model.parameters() if p.requires_grad]

    def load_checkpoint(self, path, adapter="student"):
        state=torch.load(path,map_location="cpu",weights_only=True)
        self.set_state(state["adapter"],adapter)
        return state

    def save_checkpoint(self, path, **fields):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".tmp")
        torch.save({"adapter":{k:v.detach().cpu().clone() for k,v in self.state().items()},**fields},tmp)
        os.replace(tmp,path)

    def encode(self, text, max_prompt):
        ids=self.tokenizer.apply_chat_template([{"role":"user","content":text}],
            add_generation_prompt=True,tokenize=True)
        # Explicit left truncation retained in raw record; never silently discard.
        return ids[-max_prompt:],len(ids)>max_prompt

    @torch.no_grad()
    def generate(self, prompt_ids, config, max_new, seed, deadline, adapter="teacher", exploration=0.):
        self.model.eval()
        ids=torch.tensor([prompt_ids],device=self.device)
        caches={}
        generated=[]
        generator=torch.Generator(device=self.device).manual_seed(seed)
        eos=self.tokenizer.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos or [])
        model_eos=self.model.get_base_model().generation_config.eos_token_id
        eos.update([model_eos] if isinstance(model_eos,int) else (model_eos or []))
        for _ in range(max_new):
            deadline.check()
            distributions=[]
            names=[adapter]+(["anchor"] if exploration else [])
            for current in names:
                with self.using(current):
                    out=self.model.get_base_model().model(input_ids=ids,
                        past_key_values=caches.get(current),use_cache=True)
                    caches[current]=out.past_key_values
                    distributions.append(decoder(self.logits(out.last_hidden_state[:,-1]),config))
            probs=distributions[0]
            if exploration:
                probs=(1-exploration)*probs+exploration*distributions[1]
            token=int(torch.multinomial(probs,1,generator=generator))
            generated.append(token)
            if token in eos:
                break
            ids=torch.tensor([[token]],device=self.device)
        return {"completion_ids":generated,"text":self.tokenizer.decode(generated,skip_special_tokens=True),
                "finish_reason":"eos" if generated and generated[-1] in eos else "length"}
