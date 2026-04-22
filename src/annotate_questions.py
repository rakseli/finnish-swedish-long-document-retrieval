import argparse
import json
import os
import sys
import textwrap

def load_seen_ids(args):
    seen = set()
    line_count = 0
    in_path = args.input_path
    base, _ = os.path.splitext(in_path)
    out_path = f"{base}_annotated.jsonl"
    load_seen = True
    if not os.path.exists(out_path):
        load_seen = False
    if load_seen:
        with open(out_path, "r", encoding="utf-8") as out_f:
            for line in out_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                seen.add(str(obj['id']))
       
    with open(in_path,"r") as in_f:
        for l in in_f:
            if not l:
                continue
            line_count+=1
    return seen,out_path,line_count


def prompt_label(line: str,use_translations=False) -> int:
    print("\n" + "=" * 80)
    print("\n" + "Question:")
    print(line['generated_text'])
    if use_translations:
        print("\n"+"Translation:")
        print(line['translation'])

    print("\n" + "Paragraph:")
    print(textwrap.fill(line['original_paragraph'], width=80))
    print("=" * 80)
    while True:
        ans = input("Good? [y/n/s/q] ").strip().lower()
        if ans in ("y", "yes"):
            return 1
        if ans in ("n", "no"):
            return 0
        if ans in ("q", "quit"):
            raise KeyboardInterrupt
        print("Please enter y, n, or q.")



def main():
    p = argparse.ArgumentParser(description="Annotate JSONL documents with good_text=1/0.")
    p.add_argument("--input_path", help="Path to input .jsonl file")
    args = p.parse_args()
    if "swe" in args.input_path:
        use_translations=True
    else:
        use_translations=False
    
    seen_ids,out_path,line_count = load_seen_ids(args)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(args.input_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
            for lineno, line in enumerate(fin, start=1):
                print(f"Line {lineno}",flush=True)
                line = line.strip()
                if lineno % 100 == 0:
                    print(f"Gone through {lineno/line_count*100:.2f}% of input lines...")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON on line {lineno}: {e}", file=sys.stderr)
                    continue
                if 'generated_text' not in obj:
                    print(f"Skipping line {lineno}: missing text field 'generated_text'", file=sys.stderr)
                    continue

                doc_id = str(obj['id'])
                #if the question is good
                #the id is added to seen
                #if is not, annotation goes through the docs until success
                #if annotation is stopped before the change of id, there may be duplicate annotations
                if doc_id in seen_ids:
                    continue
                label = prompt_label(obj,use_translations)
                if label is None:
                    continue
                int_label = int(label)
                obj["good_question"] = int_label
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()
                if int_label == 1:
                    seen_ids.add(doc_id)
                
    except KeyboardInterrupt:
        print("\nStopped. Progress saved to:", out_path, file=sys.stderr)


if __name__ == "__main__":
    main()