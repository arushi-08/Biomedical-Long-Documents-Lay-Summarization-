from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Any
import warnings
import gc
import random
import numpy as np
import pandas as pd
import time
import os

warnings.filterwarnings('ignore')

# Set seed for reproducibility
set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_id ='google/gemma-7b-it'

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map='auto', token=os.environ['HF_TOKEN']
# )

# model_local_id = 'models/gemma7bit_og'

# model.save_pretrained(model_local_id, from_pt=True)


# messages_eli5 = [
#     {"role": "user",
#      "content": "Summarize the following text and explain it like I'm 5 years old:\n\n{}".format(val_df['article'].iloc[idx])},
# ]

# messages_2 = [
#     {"role": "user",
#      "content": "Could you provide a lay summary of this text, ensuring the summary is highly relevant, readable, and factually accurate?\n\n{}".format(val_df['article'].iloc[idx])},
# ]

val_df = pd.read_csv('biolaysumm2024_data/eLife_val.csv')
for idx, row in val_df.iterrows():

    pipe = pipeline(
        "text-generation",
        model=model_id,
        device='cuda',
        max_new_tokens=2000,
        repetition_penalty=1.5,
    )

    if os.path.exists(f'summary/gemma7b_wo_finetuning/eLife/{idx}.txt'):
        continue
    
    message = [
        {"role": "user",
        "content": "Could you provide a lay summary of the main findings and implications of this research and how these findings are relevant to broader scientific context?\n\n{}".format(val_df['article'].iloc[idx])},
    ]

    try:
        prompt_eli5 = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        outputs_eli5 = pipe(
            prompt_eli5,
            add_special_tokens=True,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.3,
        )

        summary = outputs_eli5[0]["generated_text"][len(prompt_eli5):].replace('#', '') # 

        file_path = f'summary/gemma7b_wo_finetuning/eLife/{idx}.txt'

        # Open the file in write mode ('w') and write the long text
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(summary)

        print(f"The text for {idx} has been written to", file_path)

        time.sleep(2)
    
    except Exception as e:
        print(f"error occurred for eLIFE {idx}: ", e)


# val_df = pd.read_csv('biolaysumm2024_data/PLOS_val.csv')
# for idx, row in val_df.iterrows():

#     if os.path.exists(f'summary/gemma7b_wo_finetuning/PLOS/{idx}.txt'):
#         continue

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         device='cuda',
#         max_new_tokens=2000,
#         repetition_penalty=1.5,
#     )
#     message = [
#         {"role": "user",
#         "content": "Could you provide a lay summary of the main findings of this research and how these findings are relevant to broader scientific context?\n\n{}".format(val_df['article'].iloc[idx])},
#     ]

#     try:
#         prompt_eli5 = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
#         outputs_eli5 = pipe(
#             prompt_eli5,
#             add_special_tokens=True,
#             do_sample=True,
#             temperature=0.7,
#             top_k=20,
#             top_p=0.3
#         )

#         summary = outputs_eli5[0]["generated_text"][len(prompt_eli5):].replace('#', '') # 

#         file_path = f'summary/gemma7b_wo_finetuning/PLOS/{idx}.txt'

#         # Open the file in write mode ('w') and write the long text
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write(summary)

#         print(f"The text for {idx} has been written to", file_path)

#         time.sleep(2)

#     except Exception as e:
#         print(f"error occurred for PLOS {idx}: ", e)


print("\nValidation data inference complete")