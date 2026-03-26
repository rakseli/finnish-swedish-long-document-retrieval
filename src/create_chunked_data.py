import pandas as pd
import json
import os
import sys
from filter_corpus import serialize_datetime
from basic_tokenizer import basic_tokenize
from argparse import ArgumentParser

MAX_TOKENS = 32_000
MIN_TOKENS = 1000

def split_row_into_chunks(row, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    toks = basic_tokenize(row["text"])
    n = len(toks)
    if n <= max_tokens:
        return [row]
    chunks = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= max_tokens:
            end = n
        else:
            end = start + max_tokens
            remainder = n - end
            if 0 < remainder < min_tokens:
                end = n - min_tokens
                if end <= start:
                    end = min(start + max_tokens, n)

        tok_chunk = toks[start:end]
        text_chunk = " ".join(map(str, tok_chunk))

        new_row = {k: v for k, v in row.items() if k != "text"}
        new_row["text"] = text_chunk
        chunks.append(new_row)
        start = end

    return chunks

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--input_path',help="output file path for results")
    return ap

if __name__ == "__main__":
    args = argparser().parse_args()
    base_dir = os.path.dirname(args.input_path)
    base_file = os.path.basename(args.input_path).split(".")[0]
    output_path = os.path.join(base_dir,f"{base_file}_chunked.jsonl")
    with open(args.input_path) as in_file, open(output_path,"w") as out_file:
        for l in in_file:
            chunks = split_row_into_chunks(json.loads(l), MAX_TOKENS)
            for c in chunks:
                json_line = json.dumps(c,ensure_ascii=False,default=serialize_datetime)
                out_file.write(json_line + '\n')


            