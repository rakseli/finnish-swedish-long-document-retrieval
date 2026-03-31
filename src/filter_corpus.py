import time
import argparse
import os
import json
import io
import zstandard as zstd
import glob
from datetime import datetime
from argparse import ArgumentParser


def read_zst_files(file_path):
    if isinstance(file_path,list):
        for f_p in file_path:
            with open(f_p, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8',errors='ignore') 
                for line in text_stream:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error {e} in loading json line in file: {file_path}")
                        continue
    else:
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8',errors='ignore') 
            for line in text_stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error {e} in loading json line in file: {file_path}")
                    continue


def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")


def load_ids(args):
    if args.mode == 'corpus':
        ending = "*corpus_ids.jsonl"
    elif args.mode =='test_dev':
        ending = "*test_dev_ids.jsonl"
    else:
        raise ValueError(f"--mode should be corpus or test_dev, {args.mode} given")
    ids = set()
    f_p = glob.glob(os.path.join(args.input_root,ending))[0]
    with open(f_p,"r") as f:
        for l in f:
            j_l = json.loads(l)
            ids.add(j_l['id'])
    return ids

def filter_data(args,ids):
    count=0
    if args.mode =='corpus':
        ending = "100k_long_doc_corpus"
    else:
        ending = "test_dev_long_doc_subset"

    data_files = glob.glob(os.path.join(args.input_root,"*zst"))
    output_data_path = os.path.join(args.input_root,f"{os.path.basename(args.input_root)}_{ending}.jsonl")
    with open(output_data_path,"w") as out_file:
        for jsonl in read_zst_files(data_files):
            if jsonl['id'] in ids:
                json_line = json.dumps(jsonl,ensure_ascii=False,default=serialize_datetime)
                out_file.write(json_line + '\n')
                count+=1
                if count % 100 == 0:
                    print(f"Found {(count/len(ids))*100:.2f}% of ids",flush=True)
                if args.test:
                    if count>100:
                        break
                if count==len(ids):
                    print("All ids found, stopping...",flush=True)
                    break


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--input_root',help="output file path for results")
    ap.add_argument('--mode',default='corpus',help="filter mode, corpus for full corpus and test_dev for test data")
    ap.add_argument('--test',action='store_true')
    return ap

if __name__ == "__main__":
    args = argparser().parse_args()
    start = time.perf_counter()
    ids = load_ids(args)
    elapsed = time.perf_counter() - start
    print(f"Loaded ids: {elapsed/60:.2f} minutes",flush=True)
    start = time.perf_counter()
    filter_data(args,ids)
    elapsed = time.perf_counter() - start
    print(f"Filtered data: {elapsed/60:.2f} minutes",flush=True)
    