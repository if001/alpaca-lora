import torch
from peft import PeftModel
import transformers
import gradio as gr
import sys

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# base_model_name = "bigscience/bloom-560m"
base_model_name = "bigscience/bloom-1b1"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, add_eos_token=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model_name = sys.argv[-1]
model = PeftModel.from_pretrained(
    model,
    model_name,
    torch_dtype=torch.float16,   
)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def generate_prompt_ja(instruction, input=None):
    if input:
        return f"""以下は、タスクを説明する命令と、さらなるコンテキストを提供する入力の組み合わせです。要求を適切に満たすような応答を書きなさい。

### 命令:
{instruction}

### 入力:
{input}

### 応答:"""
    else:
        return f"""以下は、ある作業を記述した指示です。要求を適切に満たすような応答を書きなさい。

### 命令:
{instruction}

### 応答:"""


model.eval()


def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = generate_prompt_ja(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # return output.split("### Response:")[1].strip()
    return output.split("### 応答:")[1].strip()

if __name__ == "__main__":
    texts = [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]
    texts = [
        "alpacaについて教えてください",
        "2019年のメキシコの大統領を教えてください",
    ]

    # testing code for readme
    for instruction in texts:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    