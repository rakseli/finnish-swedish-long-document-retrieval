import time
import glob
import os
import json
import argparse
import torch
import sys
import re
import logging
from datasets import IterableDataset,disable_caching
from torch.utils.data import DataLoader, get_worker_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

def naive_data_collator(batch):
    """Does nothing, only for dataloader to batch samples 
    and not to convert them to tensors
    
    batch (list): list of dicts 
    Returns:
        list: list of dicts
    """    
    return batch


def data_generator(data_path):
    with open(data_path) as f:
        for l in f:
            yield json.loads(l)

def create_dataloader(args,prompt):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 if num_workers>2 else 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = IterableDataset.from_generator(lambda: data_generator(args.data_path))
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer,'prompt':prompt})
    dataloader = DataLoader(dataset,batch_size=16,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

def format_data(example,tokenizer,prompt):
    user = {"role": "user", "content":prompt.format(document_text=example["text"])}
    example['text'] = tokenizer.apply_chat_template([user],tokenize=False,reasoning_effort="low")
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="/scratch/project_2018556/models/gpt-oss-20b",type=str)
    parser.add_argument("--data_path",default="/scratch/project_2018556/finnish-swedish-long-document-retrieval/data/fin_Latn/fin_Latn_100k_long_doc_subset_chunked.jsonl",type=str,help="path to source file")
    parser.add_argument("--test",action="store_true")

    prompt = """
    Task: Split the input document into paragraphs.

Rules:
- Do NOT change any characters in the document other than inserting paragraph separators.
- The ONLY allowed change is inserting exactly two newline characters: "\n\n" between paragraphs.
- Do not delete, replace, reorder, or reformat any text. Do not fix spelling/grammar. Do not normalize whitespace.
- Preserve all existing line breaks. Do not wrap or unwrap lines.
- Do not insert paragraph breaks inside:
  - code blocks (fenced or indented)
  - tables
  - bullet/numbered lists (keep list items as-is)
- If the input already contains blank lines separating paragraphs, keep them (do not add extra blank lines).

Output format:
# Document:eos_id
<the same document text, with "\n\n" inserted only where paragraph boundaries should be>

Input document:
<<<DOCUMENT
{document_text}
DOCUMENT
>>>
    """
    args = parser.parse_args()
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(args,prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
    seed=66
    if "gpt-oss" in model_name:
        temperature = 1
        top_p = 1
        top_k = 1
        quantization="mxfp4"
        gpu_mem=0.8
        enforce_eager = False
        batch_size = 4
    else:
        raise ValueError(f"gpt-oss models should be used, {args.model_path} given")
    llm = LLM(model=args.model_path,tensor_parallel_size=2,
              max_num_seqs=batch_size,distributed_executor_backend="mp",
              disable_custom_all_reduce=True,
              max_model_len=64000,gpu_memory_utilization=gpu_mem,
              enable_chunked_prefill=True,quantization=quantization,
              enforce_eager=enforce_eager)
    
    sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                max_tokens = 64000,
                stop_token_ids=[eos_id]
                )
    total_tokens = 0
    start = time.time()
    proccessed_prompts = 0
    logger.info(f"Starting to process data")
    base_dir = os.path.dirname(args.input_path)
    with open(os.path.join(base_dir,f"{model_name}_testing_results.jsonl"),"w") as fi:
        for batch_index,batch in enumerate(dataloader,start=1):
            texts = [text_i['text'] for text_i in batch]
            ids = [s['id'] for s in batch]
            logger.info(f"Batch has {len(texts)} texts")
            logger.info(f"Running batch {batch_index}")
            outputs = llm.generate(texts, sampling_params)
            logger.info(f"Done batch {batch_index}")
            # compute throughput
            b_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
            total_tokens+=b_tokens
            for output,text_id in zip(outputs,ids):
                generated_text = output.outputs[0].text
                d = {"generated_text":generated_text,'id':text_id}
                json_line = json.dumps(d,ensure_ascii=False)
                fi.write(json_line + '\n')
                
            proccessed_prompts+=len(texts)
            if args.test:
                if batch_index==1:
                    logger.debug(f"Exiting the loop as args.test = {args.test}")
                    break
    elapsed = time.time() - start
    
    logger.info(f"Total prompts processed: {proccessed_prompts}")
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Tokens throughput: {total_tokens / elapsed:.2f} tokens/s")
    logger.info(f"Prompts throughput: {proccessed_prompts / elapsed:.2f} prompts/s")
    logger.info(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")