import json
import sys
import os
import argparse
import re
from basic_tokenizer import basic_tokenize
from lxml import etree
from filter_corpus import serialize_datetime
from statistics import mean

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)     # collapse runs of spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)  # collapse many blank lines
    return s.strip()


def split_with_tags(xml_str):
    root = etree.fromstring(xml_str.encode("utf-8"))
    paras = root.xpath("//p")
    paragraphs = []
    for p in paras:
        txt = "".join(p.itertext()).strip()
        if txt:
            paragraphs.append(txt)

    return paragraphs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path',type=str)
    args = ap.parse_args()
    base_dir = os.path.dirname(args.input_path)
    base_file = os.path.basename(args.input_path).split(".")[0]
    output_path = os.path.join(base_dir,f"{base_file}_splitted_paragraphs.jsonl")
    with open(args.input_path) as in_file, open(output_path,"w") as out_file:
        for l in in_file:
            j_l = json.loads(l)
            if j_l['good_text']!=1:
                continue
            try:
                xlm_str = j_l['xml']
                paras = split_with_tags(xlm_str)
            except KeyError as e:
                print("No xml field in j")
                text = j_l['text']
                text = normalize(text)
                full_text_tokens = len(basic_tokenize(text))
                paras = text.split("\n\n")
                para_lens = [len(basic_tokenize(p)) for p in paras]
                #if mean paragrahp is 25% of document or means are less than 100 tokens, consider it as bad splitting
                #and return none
                if (mean(para_lens)>=0.25 *full_text_tokens) or (mean(para_lens)<100):
                    print(f"Assigning paras to None as paragrahs are too long: {mean(para_lens)}")
                    paras = None
        
            j_l['paragraphs']=paras
            json_line = json.dumps(j_l,ensure_ascii=False,default=serialize_datetime)
            out_file.write(json_line + '\n')
      


