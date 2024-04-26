#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd

import os

from langchain.schema import Document
from langchain.document_loaders import JSONLoader
from langchain.chains import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

from sklearn.cluster import KMeans

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, T5Tokenizer, LongT5EncoderModel, pipeline
import tiktoken


from accelerate import PartialState


# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
# else:
#     torch.set_default_device("cpu")

model_id = "google/long-t5-local-base"


tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model = LongT5ForConditionalGeneration.from_pretrained(
    "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
).cuda()


# In[13]:


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[14]:





# In[15]:


def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["lay_summary"] = record.get("lay_summary")

    return metadata




# In[16]:


def load_json():
    # Load the pdf file
    loader = JSONLoader(
        file_path="./biolaysumm2024_data/eLife_val.jsonl",
        jq_schema='.',
        content_key="article",
        metadata_func=metadata_func,
        json_lines=True
    )

    documents = loader.load()

    token_count = num_tokens_from_string(str(documents), "cl100k_base")
    print(f'JSON Token Count: {token_count}')
    return documents, token_count




def get_embeddings(doc_splits, extractor):
    document_embeddings = []
    for doc in doc_splits:
        embedd = extractor(doc.page_content)
        mean_embedd = np.array(embedd).mean(axis=1).squeeze(axis=0)
        document_embeddings.append(mean_embedd)

    return document_embeddings
    


# In[73]:


def cluster_embeddings(embeddings, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)

    # Create an empty list that will hold your closest points
    closest_indices = []
    
    # Loop through the number of clusters you have
    for i in range(k):
        
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[i], axis=1)
        
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        
        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    return sorted(closest_indices)


# In[74]:


def summ_stage1(selected_docs, map_chain):
    # Make an empty list to hold your summaries
    summary_list = []
    
    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        
        # Append that summary to your list
        summary_list.append(chunk_summary)
        

    return summary_list


# In[75]:


def summ_stage2(summary_list, reduce_chain):
    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)
    output = reduce_chain.run([summaries])
    return output


# In[79]:


def run_long_T5_summarization(model, map_prompt, combine_prompt):
    docs, counts = load_json()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
    long_t5_enc_model = LongT5EncoderModel.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps").to("cuda")
    
    extractor = pipeline(
        model=long_t5_enc_model,
        tokenizer=tokenizer,
        task="feature-extraction",
        device="cuda"
    )
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="summarization",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=4,
        repetition_penalty=1.1,
        min_length=300,
        max_length=600,
        temperature = 0.75,
        do_sample=True,
        device="cuda"
    )
    llm_stage1 = HuggingFacePipeline(pipeline=text_generation_pipeline)
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    map_chain = load_summarize_chain(llm=llm_stage1,
                                     chain_type="stuff",
                                     prompt=map_prompt_template)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="summarization",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=4,
        min_length=300,
        max_length=400,
        temperature = 0.75,
        do_sample=True,
        device="cuda"
    )
    
    llm_stage2 = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    reduce_chain = load_summarize_chain(llm=llm_stage2,
                                 chain_type="stuff",
                                 prompt=combine_prompt_template)

    all_summ = []
    
    with open("elife.txt", "w") as f:
        doc_ids = [3, 5, 11, 14, 30, 34, 39, 44, 55, 60, 81, 82, 86, 89, 91, 95, 98, 99, 100, 103, 106, 109, 112, 121, 122, 127, 128, 136, 137, 141, 142, 
        143, 144, 158, 172, 176, 184, 188, 192, 195, 196, 214, 221, 224, 232, 234, 238]
        docs = [docs[id] for id in doc_ids]
        for doc in tqdm(docs):
            splits = text_splitter.create_documents([doc.page_content])
            embeddings = get_embeddings(splits, extractor)
            k = 5
            
            if len(splits) < 5:
                k = len(splits) - 1
            idx_for_summ = cluster_embeddings(embeddings, k=k)
    
            selected_docs = [splits[doc] for doc in idx_for_summ]
            
            summary_stage1 = summ_stage1(selected_docs, map_chain)
            
            summary_stage2 = summ_stage2(summary_stage1, reduce_chain)

            f.write(f"{summary_stage2}\n")
            f.flush()
        
        
        


# In[80]:


map_prompt = """
Summarize:
```{text}```
"""

reduce_prompt = """
Summarize:
```{text}```
"""


# In[ ]:


run_long_T5_summarization(model, map_prompt, reduce_prompt)


# In[ ]:





# In[ ]:




