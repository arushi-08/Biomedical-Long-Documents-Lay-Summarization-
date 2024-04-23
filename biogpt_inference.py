from transformers import pipeline, set_seed
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


dataset = 'eLIFE' # PLOS
model ="microsoft/biogpt" # "microsoft/BioGPT-Large"

if not os.path.exists('summary'):
    os.makedir('summary')
    os.makedir(f'summary/eLIFE')
    os.makedir(f'summary/PLOS')

val_df = pd.read_json(f'biolaysumm2024_data/{dataset}_val.jsonl', lines=True)

for idx, row in val_df.iterrows():
    if os.path.exists(f'summary/{dataset}/{idx}.txt'):
        continue

    pipe = pipeline(
        "text-generation",
        model=model,
        device='cuda',
        max_new_tokens=2000,
        repetition_penalty=1.5,
    )
    message = [
        {"role": "user",
        "content": "Could you provide a lay summary of the main findings of this research and how these findings are relevant to broader scientific context?\n\n{}".format(val_df['article'].iloc[idx])},
    ]

    try:
        prompt_eli5 = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        outputs_eli5 = pipe(
            prompt_eli5,
            add_special_tokens=True,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.3
        )

        summary = outputs_eli5[0]["generated_text"][len(prompt_eli5):].replace('#', '') # 

        file_path = f'summary/{dataset}/{idx}.txt'

        # Open the file in write mode ('w') and write the long text
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(summary)

        print(f"The text for {idx} has been written to", file_path)

        time.sleep(2)

    except Exception as e:
        print(f"error occurred for {dataset} {idx}: ", e)

    break


print("\nValidation data inference complete")