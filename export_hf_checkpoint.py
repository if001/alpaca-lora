import os
import sys
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

# BASE_MODEL = os.environ.get("BASE_MODEL", None)
# assert (
#     BASE_MODEL
# ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

"""
python export_hf_checkpoint.py \
"decapoda-research/llama-7b-hf" \
"/content/drive/MyDrive/models/alpaca_lora_ja_7b" \
"/content/drive/MyDrive/models/alpaca_lora_ja_7b/hf_ckpt"
"""

print(sys.argv)

BASE_MODEL=sys.argv[1]
LOAD_MODEL=sys.argv[2]
SAVE_DIR=sys.argv[3]

print(f"base model: {BASE_MODEL}")
print(f"load weight: {LOAD_MODEL}")
print(f"save dir: {SAVE_DIR}")


tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

# base_model = LlamaForCausalLM.from_pretrained(
#     BASE_MODEL,
#     load_in_8bit=False,
#     torch_dtype=torch.float16,
#     device_map={"": "cpu"},
# )
print("load model...")

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
offload_folder='offload'

if device == 'cpu':    
    device_map = {"": "cpu"}
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        load_in_8bit=True,
        offload_folder=offload_folder    
    )
else:
    device_map = "auto"
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map,    
        load_in_8bit=True,
        offload_folder=offload_folder
    )

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

if device == 'cpu':
    lora_model = PeftModel.from_pretrained(
        base_model,
        LOAD_MODEL,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    )   
else:
    lora_model = PeftModel.from_pretrained(
        base_model,
        LOAD_MODEL,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=True,        
    )

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
# assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, SAVE_DIR, state_dict=deloreanized_sd, max_shard_size="400MB"
)
