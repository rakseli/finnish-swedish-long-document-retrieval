import json
import sys
import os
import argparse
import re
from basic_tokenizer import basic_tokenize
from lxml import etree
from filter_corpus import serialize_datetime
from statistics import mean, median

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


def drop_if_dominant(xs, threshold):
    if xs is None or len(xs) == 0:
        return None
    total = sum(x[1] for x in xs)
    lenghts = [x[1] for x in xs]
    if total <= 0:
        return None

    if max(lenghts) / total >= threshold:
        return None

    return xs

def split_blocks_heading_join(xml_str, lb_as_newline=True):
    root = etree.fromstring(xml_str.encode("utf-8"))

    # Only take top-level-ish block nodes; list is merged, items are not selected directly
    nodes = root.xpath("//head | //p | //list")

    def normalize_keep_newlines(txt: str) -> str:
        return "\n".join(" ".join(line.split()) for line in txt.splitlines()).strip()

    def elem_text(e):
        if lb_as_newline:
            for lb in e.xpath(".//lb"):
                lb.tail = ("\n" + (lb.tail or ""))
        return "".join(e.itertext())

    def list_text(list_elem):
        lines = []
        for it in list_elem.xpath("./item"):
            t = normalize_keep_newlines(elem_text(it))
            if t:
                lines.append(t)
        return "\n".join(f"- {line}" for line in lines).strip()

    chunks = []
    pending_head = None

    for n in nodes:
        if n.tag == "head":
            pending_head = normalize_keep_newlines(elem_text(n))
            continue

        if n.tag == "list":
            body = list_text(n)
        else:  # p
            body = normalize_keep_newlines(elem_text(n))

        if not body:
            continue

        if pending_head:
            body = f"{pending_head}\n{body}"
            pending_head = None

        chunks.append(body)

    # If the XML ends with a <head> that has no following block, keep it (optional)
    if pending_head:
        chunks.append(pending_head)

    return chunks

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path',type=str)
    args = ap.parse_args()
    base_dir = os.path.dirname(args.input_path)
    base_file = os.path.basename(args.input_path).split(".")[0]
    output_path = os.path.join(base_dir,f"{base_file}_splitted_paragraphs.jsonl")
    output_stats_path = os.path.join(base_dir,f"{base_file}_splitted_paragraphs_stats.json") 
    all_paragraph_stats = []
    with open(args.input_path) as in_file, open(output_path,"w") as out_file:
        for l in in_file:
            j_l = json.loads(l)
            if j_l['good_text']!=1:
                continue
            try:
                text = j_l['xml']
                paras = split_blocks_heading_join(text)
                if len(paras) == 0:
                    text = j_l['text']
                    text = normalize(text)            
                    paras = text.split("\n\n")
            except KeyError as e:
                print("No xml field in j")
                text = j_l['text']
                text = normalize(text)            
                paras = text.split("\n\n")

            if len(paras)<2:
                print("Text, coudn't be split into paragraphs, returning None")
                paras = None
            else:
                para_lens = [len(basic_tokenize(p)) for p in paras]
                #filter short paragraphs out
                filtered_paras = [(p, l) for p, l in zip(paras, para_lens) if l > 100]
                # filter away segments that most likely are not paragraphs
                # meaning segment occupy 80% of full document
                filtered_paras = drop_if_dominant(filtered_paras,0.8)
                if filtered_paras:
                    all_paragraph_stats.extend([item[1] for item in filtered_paras])
                    paras = list(map(lambda x: x[0],filtered_paras))
                else:
                    print("Text splitting was too sparse, returning None")
   
            
            j_l['paragraphs']=paras
            try:
                json_line = json.dumps(j_l,ensure_ascii=False,default=serialize_datetime)
            except Exception as e:
                print(e)
                print(j_l)
            out_file.write(json_line + '\n')

output_stats = {"max_para_len":max(all_paragraph_stats),
                "min_para_len":min(all_paragraph_stats),
                "mean_para_len":mean(all_paragraph_stats),
                "median_para_len":median(all_paragraph_stats)}  

for k,v in output_stats.items():
    print(f"{k}: {v}")

with open(output_stats_path,"w") as output_stats_file:
    json.dump(output_stats,output_stats_file,ensure_ascii=False)


