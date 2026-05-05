from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import argparse
from glob import glob

def sifmt(i):
    # apply only to numeric cells; leave strings/NaN as-is (or NaN -> "")
    if pd.isna(i):
        return i
    if isinstance(i, (int, float)):
        affix = iter(['', 'K', 'M', 'G', 'T', 'P', 'E'])
        while i > 1000:
            i /= 1000
            next(affix)
        return f'{i:.1f}{next(affix)}'
    else:
        return i


model_map = {
   "finnish-modernbert-large-3e-05":"Finnish-ModernBERT-large",
    "finnish-modernbert-base-8e-05":"Finnish-ModernBERT-base",
    "finnish-modernbert-tiny-0.0001":"Finnish-ModernBERT-tiny",
    "xlm-roberta-large-2e-05":"XLM-RoBERTa-large",
    "mmBERT-base-1e-05":"mmBERT-base",
    "finnish-modernbert-tiny-short-8e-05":"Finnish-ModernBERT-tiny-short",
    "finnish-modernbert-base-short-0.0001":"Finnish-ModernBERT-base-short",
    "finnish-modernbert-large-short-5e-05":"Finnish-ModernBERT-large-short",
    "finnish-modernbert-tiny-short-cpt-5e-05":"Finnish-ModernBERT-tiny-short-cpt",
    "finnish-modernbert-base-short-cpt-0.0001":"Finnish-ModernBERT-base-short-cpt",
    "finnish-modernbert-large-short-cpt-2e-05":"Finnish-ModernBERT-large-short-cpt",
    "finnish-modernbert-tiny-short-edu-8e-05":"Finnish-ModernBERT-tiny-short-edu",
    "finnish-modernbert-large-short-edu-5e-05":"Finnish-ModernBERT-large-short-edu",
    "finnish-modernbert-base-short-edu-3e-05":"Finnish-ModernBERT-base-short-edu",
    "finnish-modernbert-tiny-edu-8e-05":"Finnish-ModernBERT-tiny-edu",
    "finnish-modernbert-large-edu-3e-05":"Finnish-ModernBERT-large-edu",
    "finnish-modernbert-base-edu-0.0001":"Finnish-ModernBERT-base-edu",
    "qwen-embedding-0.6b":"Qwen3-Embedding-0.6B"
    

}

qwen_results =  [
    {"model":"Qwen3-Embedding-0.6B",
    "language":"swe",
    "main_score":0.30016},
    {"model":"Qwen3-Embedding-0.6B",
    "language":"fin",
    "main_score":0.23656}
]

def make_latex_table_from_runs(
    root_dir: str | Path,
    *,
    task_prefix: str = "FinnishSwedishLongDocRetrieval",
    split: str = "test",                  # typically "test"
    languages: Iterable[str] = ("fin", "swe"),
    json_suffix: str = "_test.json",      # files like ..._fin_test.json, ..._swe_test.json
    sort_by: tuple[str, ...] = ("model", "language"),
    caption: Optional[str] = "Model performance in out-of-domain retrieval FSLDR using the nDCG@10 metric",
    label: Optional[str] = "tab:fsldr",
    float_format: str = "%.3f",
) -> str:
    """
    Scans `root_dir` for per-run folders (each run folder = one model setting).
    From each run folder, reads files like:
      {task_prefix}_{lang}{json_suffix}
    and extracts:
      data["task_results"][0]["scores"][split][0]["main_score"]

    Returns a LaTeX table string.
    """
    root_dir = Path(root_dir)

    rows = []
    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        model_name = run_dir.name

        for lang in languages:
            # expected: FinnishSwedishLongDocRetrieval_fin_test.json, etc.
            json_path = run_dir / f"{task_prefix}_{lang}{json_suffix}"
            if not json_path.exists():
                # silently skip missing combinations
                continue

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            try:
                main_score = (
                    data["task_results"][0]["scores"][split][0]["main_score"]
                )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(
                    f"Could not extract main_score from {json_path}. "
                    f"Expected data['task_results'][0]['scores']['{split}'][0]['main_score']."
                ) from e

            rows.append(
                {
                    "model": model_map[model_name],
                    "language": lang,
                    "main_score": float(main_score),
                }
            )
    #add qwen results manually
    rows.extend(qwen_results)
    if not rows:
        raise ValueError(f"No results found under {root_dir}")

    df = pd.DataFrame(rows)
    if sort_by:
        df = df.sort_values(list(sort_by), kind="stable")

    # Wide format: one row per model, columns per language (fin/swe)
    wide = df.pivot(index="model", columns="language", values="main_score")
    wide = wide.reindex(columns=list(languages))  # ensure column order

    # Optional: add an average column
    # wide["avg"] = wide.mean(axis=1)

    latex = wide.to_latex(
        index=True,
        na_rep="--",
        float_format=float_format % 0.0 if "% " in float_format else (lambda x: float_format % x)
        if "%" in float_format else None,
        caption=caption,
        label=label,
        escape=False,
    )
    return latex

def make_corpus_stats_table(root_dir,
    caption: Optional[str] = "Corpus statistics.",
    label: Optional[str] = "tab:corpus_stats",
) -> str:
    files = glob(f"{root_dir}/**/*_corpus_stats.csv")
    if "fin" in files[0]:
        fin_file_path = files[0]
        swe_file_path = files[1]

    else:
        fin_file_path = files[1]
        swe_file_path = files[0]
    fi_df = pd.read_csv(fin_file_path,header=0,names=['Statistic', 'Tokens',"Characters"])
    fi_df["Language"] = "Fin"
    swe_df = pd.read_csv(swe_file_path,header=0,names=['Statistic', 'Tokens',"Characters"])
    swe_df["Language"] = "Swe"
    combined = pd.concat([fi_df, swe_df], ignore_index=True)
    wide = combined.pivot(index="Statistic", columns="Language", values=["Tokens", "Characters"])
    wide = wide.map(sifmt)
    order = ['count', 'total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    wide = wide.reindex(order)
    latex = wide.to_latex(
        index=True,
        caption=caption,
        label=label,
    )
    return latex


def make_paragrahp_stats_table(root_dir,
    caption: Optional[str] = "Corpus statistics.",
    label: Optional[str] = "tab:corpus_stats",
) -> str:
    files = glob(f"{root_dir}/**/*_paragraph_start_stats.csv")
    if "fin" in files[0]:
        fin_file_path = files[0]
        swe_file_path = files[1]
    else:
        fin_file_path = files[1]
        swe_file_path = files[0]

    fi_df = pd.read_csv(fin_file_path,header=None,names=['Statistic', 'Tokens'])
    fi_df["Language"] = "Fin"
    swe_df = pd.read_csv(swe_file_path,header=None,names=['Statistic', 'Tokens'])
    swe_df["Language"] = "Swe"
    combined = pd.concat([fi_df, swe_df], ignore_index=True)
    wide = combined.pivot(index="Statistic", columns="Language", values=["Tokens"])
    wide = wide.map(sifmt)
    order = ['count', 'total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    wide = wide.reindex(order)
    latex = wide.to_latex(
        index=True,
        caption=caption,
        label=label,
    )
    return latex


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_path',default=None,help="path for run results")
    args = ap.parse_args()
    if "results/mteb_evaluations" in args.root_path:
        table = make_latex_table_from_runs(root_dir=args.root_path)
        print(table)
    else:
        table = make_corpus_stats_table(args.root_path)
        print(table)
        print()
        table = make_paragrahp_stats_table(args.root_path)
        print(table)
