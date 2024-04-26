import os
import torch
import pandas as pd
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState

model_id = "google/gemma-7b-it"
model_type = "gemma7bit_og"

from datasets import Dataset

df = pd.read_json('biolaysumm2024_data/eLife_train.jsonl', lines=True).iloc[0]
val_df = pd.read_json('biolaysumm2024_data/eLife_val.jsonl', lines=True).iloc[0]

data = Dataset.from_pandas(df[['lay_summary', 'article']])
val_data = Dataset.from_pandas(val_df[['lay_summary', 'article']])

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16
)

# os.environ['HF_HOME'] = '/data_vault/hexai/huggingface/hub/'

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map='auto', token=os.environ['HF_TOKEN']
)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
def formatting_func(example):
    output_texts = []
    for i in range(len(example['article'])):
        messages = [
            {"role": "user",
             "content": f"""
                Summarize this document. Text: {example['article'][i]}. 
                Summary:
                """},
             {"role": "assistant",
             "content": "{}".format(example['lay_summary'][i])}
         ]
        output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return output_texts


import transformers
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    eval_dataset=val_data,
    max_seq_length=2000,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        # max_steps=50,
        num_train_epochs=1,
        # eval_steps=5, 
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
	evaluation_strategy="steps",     # evaluate each `eval_steps`
    	eval_steps=500,                  # number of steps to run evaluation
    	save_strategy="steps",
    	save_steps=500,
    	load_best_model_at_end=True
    ),
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
    peft_config=lora_config,
    formatting_func=formatting_func
)

# model = trainer.model.merge_and_unload()
trainer.save_model(f'models/{model_type}')

# peft_model_id = "Model/blip-saved-model"
# config = PeftConfig.from_pretrained(peft_model_id)

