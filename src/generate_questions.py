import time
_GENERATION_START_TIME = time.time()
import glob
import os
import json
import argparse
import torch
import sys
import re
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
import numpy as np
from datasets import IterableDataset,disable_caching
from torch.utils.data import DataLoader, get_worker_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datetime import datetime

logger = logging.getLogger(__name__)

generic_system_prompt="""You are a helpful and focused AI assistant.

    Always follow the user’s instructions carefully and complete the requested tasks to the best of your ability.
    Provide clear, accurate, and relevant responses that stay on topic.
    Do not include extra information or content that was not requested by the user.

    """

def naive_data_collator(batch):
    """Does nothing, only for dataloader to batch samples 
    and not to convert them to tensors
    
    batch (list): list of dicts 
    Returns:
        list: list of dicts
    """    
    return batch


def data_generator(data_path,seed):
    rng = np.random.default_rng(seed)
    with open(data_path) as f:
        for l in f:
            j_l = json.loads(l)
            if j_l['good_text']!=1:
                continue
            if j_l['paragraphs'] is None:
                continue
            if len(j_l['paragraphs'])>5:
                paras = rng.choice(j_l['paragraphs'],5)
            else:
                paras = j_l['paragraphs']
                rng.shuffle(paras)
            for p in paras:
                yield {'text':p,'id':j_l['id']}

def create_dataloader(args,prompt,seed):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 if num_workers>2 else 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.lang == "fin":
        pronouns = "tässä / siinä"
        language = "Finnish"
    elif args.lang == "swe":
        pronouns = "här / det här"
        language = "Swedish"
    else:
        raise ValueError("fin or swe should be used as languages")
    dataset = IterableDataset.from_generator(lambda: data_generator(args.data_path,seed=seed))
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer,'prompt':prompt,'lang':language,'pronouns':pronouns})
    dataloader = DataLoader(dataset,batch_size=100,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

def format_data(example,tokenizer,prompt,lang,pronouns):
    user = {"role": "user", "content":prompt.format(text=example['text'],lang=lang,pronouns=pronouns)}
    system = {"role":"system","content":generic_system_prompt}
    example['paragraph']=example['text']
    example['text'] = tokenizer.apply_chat_template([system,user],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return example

def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def detect_completed_batches(out_path, batch_size):
    """
    Returns number of *completed* batches already present in jsonl.
    We assume you write exactly one json line per sample.
    Only full batches are considered completed; a partial last batch will be re-run.
    """
    if not os.path.exists(out_path):
        return 0
    n_lines = count_lines(out_path)
    return n_lines // batch_size

def skip_batches(dataloader, n_to_skip):
    """
    Advance the iterator by n_to_skip batches.
    """
    if n_to_skip == 0:
        return dataloader

    it = iter(dataloader)
    for _ in range(n_to_skip):
        next(it)
    return it

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="/scratch/project_2018556/models/Qwen3.5-122B-A10B-GPTQ-Int4",type=str)
    parser.add_argument("--data_path",default="/scratch/project_2018556/finnish-swedish-long-document-retrieval/data/swe_Latn/swe_Latn_test_dev_long_doc_subset_annotated_splitted_paragraphs.jsonl",type=str,help="path to source file")
    parser.add_argument("--exit_duration_in_mins",type=int, default=None, help="exit duration")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--lang",default="fin")
    
    prompt = """
Task:
Generate one specific and valuable question in {lang} based on the following text. 
The generated question should revolve around the core content of this text, and avoid using pronouns like "{pronouns}" 

Rules:
- The question should be answerable only using the text
- Generate only one question in {lang}

Output format:
# Question:
<the generated question>

Input text:
<<<text
{text}
text
>>>
    """
    SEED = 66
    args = parser.parse_args()
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(args,prompt,SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
    prompt_len = len(tokenizer(prompt).input_ids)
    logger.info(f"Prompt lenght in tokens: {prompt_len}")
    del tokenizer
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p=0.0
    gpu_mem=0.9
    enforce_eager = False
    presence_penalty=1.5
    repetition_penalty=1.0
    batch_size = 16
    max_model_len = 72000
    tensor_parallel_size = 4
    llm = LLM(model=args.model_path,tensor_parallel_size=tensor_parallel_size,
              max_num_seqs=batch_size,distributed_executor_backend="mp",
              disable_custom_all_reduce=True,
              max_model_len=max_model_len,gpu_memory_utilization=gpu_mem,
              enable_chunked_prefill=True,
              enforce_eager=enforce_eager,
              language_model_only=True
              )

    sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=12000,
                seed=SEED,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[eos_id]
                )
    total_tokens = 0
    start = time.time()
    proccessed_prompts = 0
    logger.info(f"Starting to process data")
    base_dir = os.path.dirname(args.data_path)
    base_name = os.path.basename(args.data_path).split(".")[0]
    if args.test:
        base_name = "test_run"
    # Determine where to start (batch index is 1-based)
    out_path = os.path.join(base_dir,f"{base_name}_{model_name}_generate_questions.jsonl")
    batches_to_skip = 0
    # Auto-detect based on existing file length
    dataloader_batch_size = dataloader.batch_size
    batches_to_skip = detect_completed_batches(out_path, dataloader_batch_size)
    logger.info(f"Output: {out_path}")
    logger.info(f"Resuming: skip {batches_to_skip} batches")
    # Open file in append mode if resuming and file exists, else write mode
    file_mode = "a" if (batches_to_skip > 0 and os.path.exists(out_path)) else "w"
    start = time.time()
    total_tokens = 0
    proccessed_prompts = 0
    logger.info("Starting to process data")
    all_generated = True

    with open(out_path, file_mode, encoding="utf-8") as fi:
        # Create an iterator and skip batches
        dl_iter = skip_batches(dataloader, batches_to_skip)
        for batch_index,batch in enumerate(dl_iter,start=1):
            elapsed_time = (time.time() - _GENERATION_START_TIME) / 60.0
            if elapsed_time > args.exit_duration_in_mins:
                logger.info('Exiting program gracefylly after {} minutes'.format(elapsed_time),flush=True)
                all_generated = False
                break
            texts = [text_i['text'] for text_i in batch]
            paragraphs = [text_i['paragraph'] for text_i in batch]
            ids = [s['id'] for s in batch]
            logger.info(f"Batch has {len(texts)} texts")
            logger.info(f"Running batch {batch_index}")
            outputs = llm.generate(texts, sampling_params)
            logger.info(f"Done batch {batch_index}")
            # compute throughput
            b_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
            total_tokens+=b_tokens
            for output,text_id,text,paragraph in zip(outputs,ids,texts,paragraphs):
                generated_text = output.outputs[0].text
                d = {"original_paragraph":paragraph,"prompt":text,"generated_text":generated_text,'id':text_id}
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
    if all_generated:
        logger.info("All questions were generated")
    else:
        logger.info("Generation did not exhaust")
