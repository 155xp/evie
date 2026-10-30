"""Dump a HuggingFace-transformers reference forward pass for validating evie's engine.

usage: python tools/reference.py <model_dir> <out.json> [prompt] [dtype]
"""

import json
import sys

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

model_dir, out_path = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Who are you?"
dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[sys.argv[4] if len(sys.argv) > 4 else "f32"]

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(model_dir)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).eval()
lm = model.model.language_model

msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
ids = tok(text, add_special_tokens=False).input_ids

acts, resid = {}, {}
for i, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0][0].float().clone()))
    layer.register_forward_hook(
        lambda m, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0].float().clone())
    )

logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
acts, resid = dict(acts), dict(resid)  # generate() below re-fires the hooks
for layer in lm.layers:
    layer.mlp.down_proj._forward_pre_hooks.clear()
    layer._forward_hooks.clear()
gen = model.generate(input_ids=torch.tensor([ids]), max_new_tokens=24, do_sample=False)

json.dump(
    {
        "text": text,
        "ids": ids,
        "top": [[int(i), float(logits[i])] for i in logits.topk(20).indices],
        "logits_stride": 997,
        "logits": logits[::997].tolist(),
        "resid_norms": [resid[i].norm(dim=-1).tolist() for i in range(len(lm.layers))],
        "act_stride": 61,
        "acts": [acts[i][:, ::61].tolist() for i in range(len(lm.layers))],
        "greedy": gen[0, len(ids):].tolist(),
    },
    open(out_path, "w"),
)
print("wrote", out_path)
