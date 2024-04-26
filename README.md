# LLM Summarization

### Course: CS2731 Introduction to Natural Language Processing

<hr />

### Environment setup

`conda env update --file nlp.yml --prune`
<hr />

### Run experiments with Huggingface models

Gemma LoRA finetuning: `python lora_finetune_gemma7b.py`

Gemma inference pipeline: `python gemma_inference_eli5.py`

Long T5 LoRA finetuning with BRV method:
`python brv_summarization_LongT5.py`


<hr />

### Summaries saved in summary dir.

(Not recommended) Downloads more than 10GiB of data by scraping financial website. Can be run using `download_news_data.py`
