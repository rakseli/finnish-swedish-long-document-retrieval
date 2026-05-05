import json
import os
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import ndcg_score

import evaluate
import glob
from create_latex_tables import model_map

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_query_ids_over_threshold(query_meta_jsonl: Path, token_threshold_lower: int = 16_000,token_treshold_upper: int = None):
    """
    Returns:
      set(query_id) where tokens_before_paragraph > token_threshold
    """
    selected = set()
    for obj in _read_jsonl(query_meta_jsonl):
        qid = obj.get("query-id")
        tbp = obj.get("tokens_before_paragraph")
        if token_treshold_upper is None:
            if qid is not None and tbp is not None and tbp > token_threshold_lower:
                selected.add(qid)
        if token_threshold_lower is None:
            if qid is not None and tbp is not None and tbp < token_treshold_upper:
                selected.add(qid)
        if token_threshold_lower is not None and token_treshold_upper is not None:
            if qid is not None and tbp is not None and tbp > token_threshold_lower and tbp < token_treshold_upper:
                selected.add(qid)
    return selected


def load_qrels_as_relevance_dict(qrels_jsonl: Path):
    """
    Convert qrels jsonl into:
      rel[qid][docid] = relevance_score (int/float)

    Assumes qrels line format:
      {"query-id": "...", "corpus-id": "...", "score": 1}
    """
    rel = defaultdict(dict)
    for obj in _read_jsonl(qrels_jsonl):
        qid = obj["query-id"]
        docid = obj["corpus-id"]
        score = obj.get("score", 0)
        rel[qid][docid] = score
    return rel


def load_predictions(pred_json: Path, lang: str, split: str = "test"):
    """
    Loads predictions from a single predictions.json with the structure like:
      {
        "mteb_model_meta": {...},
        "fin": {"test": {"qid": {"docid": score, ...}, ...}},
        "swe": {"test": {...}}
      }

    Returns:
      dict[qid] -> dict[docid] -> score
    """
    with pred_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if lang not in data or split not in data[lang]:
        raise KeyError(f"Missing predictions for lang={lang}, split={split} in {pred_json}")

    return data[lang][split]


def _to_evaluate_lists(preds_by_qid, qrels_by_qid, query_ids, k=10):
    """
    Build lists compatible sklearn nDCG
    We'll compute query-wise nDCG@k

    Returns:
      y_true_list, y_score_list  (lists of lists)
        - each element corresponds to one query
        - each is a list of length = number of candidate docs considered for that query
    """
    y_true_list = []
    y_score_list = []

    for qid in sorted(query_ids):
        qpred = preds_by_qid.get(qid, {})
        if not qpred:
            continue  # no predictions => skip or treat as 0; skipping is usually clearer

        # consider top-k predicted docs (by score) for this query
        top_docs = sorted(qpred.items(), key=lambda x: x[1], reverse=True)[:k]
        doc_ids = [d for d, _ in top_docs]
        scores = [s for _, s in top_docs]

        # relevance labels for those docs (0 if not in qrels)
        qrels = qrels_by_qid.get(qid, {})
        rels = [qrels.get(docid, 0) for docid in doc_ids]

        y_true_list.append(rels)
        y_score_list.append(scores)

    return y_true_list, y_score_list


def ndcg_at_10_for_long_queries(
    model_dir: str | Path,
    base_data_dir: str | Path,
    langs=("fin", "swe"),
    split="test",
    token_threshold_lower=16_000,
    token_threshold_upper=None,
    pred_filename_glob="/**_test/*predictions.json",  # tries to find your predictions.json
):
    """
    Computes mean nDCG@10 for queries where tokens_before_paragraph > token_threshold,
    separately for each language, using:
      - queries-meta: {base_data_dir}/{lang}-queries-meta/{split}.jsonl
      - qrels:       {base_data_dir}/{lang}-qrels/{split}.jsonl
      - predictions: searched inside model_dir by glob (default **/*predictions.json)

    Returns:
      dict like {"fin": {"ndcg@10": ..., "n_queries": ...}, "swe": {...}}
    """
    base_data_dir = Path(base_data_dir)
    # locate predictions.json under model_dir
    model_name = os.path.basename(model_dir)
    pred_paths = glob.glob(f"{model_dir}/**test/*.json")
    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction json found under {model_dir} with glob '{pred_filename_glob}'"
        )

    if "_fin_test" in pred_paths[0]:
        fin_preds_path = pred_paths[0]
        swe_preds_path = pred_paths[1]
    else:
        fin_preds_path = pred_paths[1]
        swe_preds_path = pred_paths[0]

    results = []
    pred_paths = [Path(fin_preds_path),Path(swe_preds_path)]
    for lang,pred_path in zip(langs,pred_paths):
        query_meta_jsonl = base_data_dir / f"{lang}-queries-meta" / f"{split}.jsonl"
        qrels_jsonl = base_data_dir / f"{lang}-qrels" / f"{split}.jsonl"
        long_qids = load_query_ids_over_threshold(query_meta_jsonl, token_threshold_lower=token_threshold_lower,token_treshold_upper=token_threshold_upper)
        qrels = load_qrels_as_relevance_dict(qrels_jsonl)
        preds = load_predictions(pred_path, lang=lang, split=split)
        y_true, y_score = _to_evaluate_lists(preds, qrels, long_qids, k=10)
        score = ndcg_score(y_true=y_true,
                           y_score=y_score,
                           k=10,
                           ignore_ties=True
                           )
        results.append({"ndcg@10": score,
                        "n_queries": len(y_true),
                        "token_treshold_upper":token_threshold_upper,
                        "token_threshold_lower":token_threshold_lower,
                        "model":model_map[model_name],
                        "language":lang}            
                    )
    return results


if __name__ == "__main__":
    base_data_dir = "/scratch/project_2018556/finnish-swedish-long-document-retrieval/results/fsldr"
    models_dir = "/scratch/project_2018556/finnish-swedish-long-document-retrieval/results/mteb_evaluations"
    all_results = []
    for model_dir in glob.glob(f"{models_dir}/*"):
        if not os.path.isdir(model_dir):
            continue
        for t in [(0,1024),(1024,8192),(8192,16000),(16000,32000),(32000,None)]:
            r = ndcg_at_10_for_long_queries(model_dir=model_dir, base_data_dir=base_data_dir,token_threshold_lower=t[0],token_threshold_upper=t[1])
            all_results.extend(r)
    
    with open(f"{models_dir}/detailed_retrieval_analysis.jsonl","w") as out_file:
        for r in all_results:
            json_line = json.dumps(r,ensure_ascii=False)
            out_file.write(json_line + '\n')
