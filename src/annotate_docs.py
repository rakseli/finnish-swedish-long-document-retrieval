import argparse
import json
import os
import sys


def load_seen_ids(out_path: str, id_field: str) -> set:
    seen = set()
    n_good_texts = 0 
    if not os.path.exists(out_path):
        return seen, n_good_texts

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj['good_text']==1:
                n_good_texts+=1
            if id_field in obj:
                seen.add(str(obj[id_field]))
    return seen, n_good_texts


def prompt_label(line: str) -> int:
    print("\n" + "=" * 80)
    print("\n" + "url:" + line['u'])
    print(line['text'][:1000])
    print("=" * 80)
    while True:
        ans = input("Good text? [y/n/q] ").strip().lower()
        if ans in ("y", "yes"):
            return 1
        if ans in ("n", "no"):
            return 0
        if ans in ("q", "quit"):
            raise KeyboardInterrupt
        print("Please enter y, n, or q.")


def main():
    p = argparse.ArgumentParser(description="Annotate JSONL documents with good_text=1/0.")
    p.add_argument("input_file", help="Path to input .jsonl file")
    p.add_argument("--text_field", default="text", help="Field name for text (default: text)")
    args = p.parse_args()

    in_path = args.input_file
    base, _ = os.path.splitext(in_path)
    out_path = f"{base}_annotated.jsonl"

    seen_ids,n_good_texts = load_seen_ids(out_path, 'id')
    if len(seen_ids) == 1500:
        print(f"All data processed, good texts: {n_good_texts}")
        sys.exit(0)
    # Append mode so we can resume.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
            for lineno, line in enumerate(fin, start=1):
                print(f"Line {lineno}",flush=True)
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON on line {lineno}: {e}", file=sys.stderr)
                    continue


                if args.text_field not in obj:
                    print(f"Skipping line {lineno}: missing text field '{args.text_field}'", file=sys.stderr)
                    continue

                doc_id = str(obj['id'])
                if doc_id in seen_ids:
                    continue  # already annotated in output

                
                label = prompt_label(obj)
                int_label = int(label)
                obj["good_text"] = int_label
                if int_label==1:
                    n_good_texts+=1
                
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()

                seen_ids.add(doc_id)
                if n_good_texts % 10 == 0:
                    print(f"Done {n_good_texts/1000*100:.2f}%",flush=True)
                if n_good_texts == 1000:
                    print("1K good texts found")
    except KeyboardInterrupt:
        print("\nStopped. Progress saved to:", out_path, file=sys.stderr)


if __name__ == "__main__":
    main()