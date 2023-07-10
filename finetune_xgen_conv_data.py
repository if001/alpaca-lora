'''
memo


モデルごとにどのtarget_moduleセットするか

https://github.com/huggingface/peft/blob/13e53fc7ee5d89d59b16523051006dddf0fb7a49/src/peft/mapping.py#L41
'''
import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, Dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils.prompter import Prompter

# OUTPUT_DIR="/content/drive/MyDrive/models/alpaca-lora-test"
# SAVE_PRE_TRAINED_DIR = "/content/drive/MyDrive/models/alpaca-lora-test/bloom-lora-ja"

"""
python finetune.py \
--base_model "decapoda-research/llama-7b-hf" \
--output_dir "/content/drive/MyDrive/models/alpaca-lora-ja"  \
--data_path "./japanese_alpaca_data.json"
--prompt_template_name "alpaca_ja"
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils import PreTrainedTokenizerBase
import numpy as np


@dataclass
class DataCollatorForSeq2SeqDebug:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.


        for v in features:
           #print('input decode', tokenizer.decode(v['input_ids']))
           #print('label decode', tokenizer.decode(v['labels']))
           print(len(v['input_ids']), len(v['labels']))
        print('--')
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            print('padding', padding_side)
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                            
        for v in features:
           #print('input decode', tokenizer.decode(v['input_ids']))
           #print('label decode', tokenizer.decode(v['labels']))
           print(len(v['input_ids']), len(v['labels']))
        

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        #print('batch input: ', len(features['input_ids']))
        #print('batch labels', len(features['labels']))
        
        # for x, y in zip(features['input_ids'], features['labels']):
        #     print('input_ids:', len(x), 'lables: ', len(y))
        # print(len(features['input_ids']), len(features['labels']))
        
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
            print('decoder_input_ids: ', len(feature['decoder_input_ids']))

        print('ids, labels: ', features['input_ids'].shape, features['labels'].shape)
        print('-'*20)
        return features


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca_ja",  # The prompt template to use, will default to alpaca.
    # max_steps: int = 4,
):
    # transformers.logging.set_verbosity_info()
    transformers.logging.set_verbosity_debug()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    # quantization_config = BitsAndBytesConfig(
    #    llm_int8_enable_fp32_cpu_offload = True,
    # )
    ## for qlora
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,        
        #load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
        #offload_folder="offload",
        #offload_state_dict = True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
                                            base_model, 
                                            trust_remote_code=True,
                                            # pad_token='<|endoftext|>'
                                            pad_token='<|pad|>'
                                        )
    # tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.padding_side = "right"  # Allow batched inference

    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)
    print('special token: ', tokenizer.special_tokens_map)
    print('pad:', tokenizer.pad_token)
    print('eos:', tokenizer.eos_token)
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        data = []
        prefix = "以下はユーザーとアシスタントの会話です。アシスタントは親切で丁寧に詳細を回答します。\n\n"
        for conversations in data_point["conversations"]:
            for i in range(len(conversations)):
                prompt = prefix
                for j, v in enumerate(conversations[:i+1]):
                    prompt += "### ユーザー: \n" + v["S"] + '\n\n' + "### アシスタント: \n" + v["U"]
                    if j != i:
                        prompt += '\n\n'
                    #print(prompt)
                    # print('-'*20)
                    tokenized_prompt = tokenize(prompt, add_eos_token=True)
                    data.append(tokenized_prompt)
        return data
    
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        print('data set len', len(data["train"]))
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        # print('train_val', train_val)
        train_data = generate_and_tokenize_prompt(train_val["train"].shuffle())
        print("train_data len", len(train_data))        
        val_data = generate_and_tokenize_prompt(train_val["test"].shuffle())
        print("val_data", len(val_data))
        print("train_data", train_data[0])
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            # max_steps=4,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2SeqDebug(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True, 
            label_pad_token_id=tokenizer.pad_token_id
        ),
        # data_collator=DataCollatorForSeq2SeqDebug(
        #     tokenizer, 
        #     pad_to_multiple_of=8,
        #     return_tensors="pt",
        #     padding=True, 
        #     label_pad_token_id=tokenizer.pad_token_id,
        # ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
