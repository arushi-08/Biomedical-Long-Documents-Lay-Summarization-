import pandas as pd

from datasets import Dataset

import os
import torch
from peft import LoraConfig, TaskType
import transformers
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from trl import SFTTrainer

df = pd.read_json('./biolaysumm2024_data/eLife_train.jsonl', lines=True)
val_df = pd.read_json('./biolaysumm2024_data/eLife_val.jsonl', lines=True)

def formatting_func_t5(examples):
    output_texts = []
    for i in range(len(examples['article'])):
        message = f"Summarize: {examples['article'][i]}"
        output_texts.append(message)
    return output_texts

data = Dataset.from_pandas(df[['lay_summary', 'article']])
val_data = Dataset.from_pandas(val_df[['lay_summary', 'article']])

model_id = "longt5-large-16384-pubmed" #"google/long-t5-local-base"

#bnb_config = BitsAndBytesConfig(
#    load_in_8bit=True,
#    bnb_8bit_quant_type="nf8",
#    bnb_8bit_compute_dtype=torch.float16
#)

tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model = LongT5ForConditionalGeneration.from_pretrained(
    "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
)

# model.tie_weights()

lora_config = LoraConfig(
 r=16, 
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    eval_dataset=val_data,
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func_t5
)
trainer.train()

# Merge the adapters into the base model so you can use the model like a normal transformers model
model = trainer.model.merge_and_unload()
model.save_pretrained(f'./models/{model_id}_lora_finetuned')