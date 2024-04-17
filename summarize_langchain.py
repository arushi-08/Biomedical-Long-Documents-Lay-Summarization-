import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.document_loaders import JSONLoader
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain
from langchain.text_splitter import TokenTextSplitter

model_type = 'gemma2b'
model_id = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(
    f"{model_type}", torch_dtype=torch.bfloat16
    )
    
os.environ['HF_TOKEN'] = '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'

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], cache_dir=os.environ['HF_HOME'])

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_length = 13000,
    trust_remote_code=True,
    device_map="auto",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs = {'temperature':0.75})

map_prompt_template = """
Write a concise lay term summary of this chunk of text from a medical article.
{text}
"""

# map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

# system_prompt =  f"Summarize in lay terms. For example\n\nArticle:{elife_train.loc[2529].article}\nLay Summary:{elife_train.loc[2529].lay_summary}" 

# combine_temp = """ 
# Write a concise lay term summary of the following text delimited by triple backquotes.
# ```{text}```
# """

# combine_prompt = PromptTemplate(
#     template=system_prompt + combine_temp, input_variables=["text"]
# )

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["lay_summary"] = record.get("lay_summary")
    return metadata


map_prompt = PromptTemplate.from_template(map_prompt_template)
map_chain = LLMChain(prompt=map_prompt, llm=llm)


reduce_template = """The following is set of summaries of a medical article:
{doc_summaries}
Summarize the above summaries. 
Summary:"""


reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
stuff_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries")


reduce_chain = ReduceDocumentsChain(
    combine_documents_chain=stuff_chain,
    token_max=8000
)

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    document_variable_name="text",
    reduce_documents_chain=reduce_chain
)

loader = JSONLoader(
    file_path="/ihome/cs2731_2024s/ars539/biolaysumm2024_data/eLife_val.jsonl",
    jq_schema='.',
    content_key="article",
    metadata_func=metadata_func,
    json_lines=True
)

data = loader.load()


splitter = TokenTextSplitter(chunk_size=500)

for i in range(len(data)):
    x = data[i]
    split_docs = splitter.split_documents([x])

    summary = map_reduce_chain.run(split_docs)

    with open(f"summaries/{i}.txt", "w") as text_file:
        text_file.write(summary)
