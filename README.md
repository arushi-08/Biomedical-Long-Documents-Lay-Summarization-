# Biomedical Long Documents Lay Summarization using Large Language Models

### Environment setup

`conda env update --file nlp.yml --prune`
<hr />

### Run experiments with Huggingface models

Gemma LoRA finetuning: `python lora_finetune_gemma7b.py` (requires GPU server with 45+ GB RAM)

Gemma inference pipeline: `python gemma_inference_eli5.py`

Long T5 LoRA Inference with BRV method:
`python brv_summarization_LongT5.py`

Long T5 LoRA finetuning: `lora_finetine_LongT5_pubmed.py`

BioGPT inference pipeline: `python biogpt_inference.py`

<hr />

Summaries saved in summary dir.
