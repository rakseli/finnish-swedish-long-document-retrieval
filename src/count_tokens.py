import json
import time
from argparse import ArgumentParser
from basic_tokenizer import basic_tokenize
from filter_corpus import read_zst_files

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--output_file',help="output file path for results")
    ap.add_argument('--input_file',help="input file path")
    ap.add_argument('--test',action='store_true')
    return ap

def count_tokens(args):
    d = []
    docs = 0
    with open(args.output_file, "w") as out_file:
        for ind,line in enumerate(read_zst_files(args.input_file),start=1):
            text = line['text']
            uid = line['id']
            docs += 1
            words = len(basic_tokenize(text))
            chars = len(text)
            data_out = {"id":uid,"tokens":words,"chars":chars}
            json_line = json.dumps(data_out,ensure_ascii=False)
            out_file.write(json_line + '\n')
            if args.test:
                if ind==1000:
                    break
if __name__ == "__main__":
    args = argparser().parse_args()
    start = time.perf_counter()
    count_tokens(args)
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.2f} seconds")