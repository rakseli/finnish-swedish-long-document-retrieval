# Adapted from https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/evaluate_st.py
# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
import argparse
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
import mteb
import sentence_transformers
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name',help="model name")
    ap.add_argument('--lr',help="lr for bookkeeping")
    ap.add_argument('--lang',help="lang to use in task")
    ap.add_argument('--split',help="split to use in task")

    args = ap.parse_args()

    all_models_dir = "/scratch/project_2018556/finetuned-ms-marco-models/sentence-transfromers"
    output_base_dir = "../results/mteb_evaluations"
    model_shortname = args.model_name
    trained_model_named = f"{model_shortname}-{args.lr}"
    base_models_dir = os.path.join(all_models_dir,model_shortname+"_msmarco")
    individual_run_dir = os.path.join(base_models_dir,trained_model_named)
    final_model_dir = os.path.join(individual_run_dir,"final")
    output_folder = os.path.join(output_base_dir,f"{trained_model_named}")
    task_name = "FinnishSwedishLongDocRetrieval"
    task_results_path = os.path.join(output_folder,f"{task_name}_{args.lang}_{args.split}.json")
    model_predictions_folder = os.path.join(output_folder,f"{task_name}_model_predictions_{args.lang}_{args.split}")
    if model_shortname != "xlm-roberta-large":
        model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}
        encode_kwargs={"batch_size": 16}
        #can handle b size 16
    else:
        model_kwargs = {"torch_dtype": "bfloat16"}
        encode_kwargs={"batch_size": 64}

    logger.info("Loading model")
    model_card_data = sentence_transformers.sentence_transformer.model_card.SentenceTransformerModelCardData(model_id=f"local-finetune/{trained_model_named}_msmarco",local_files_only=True,generate_widget_examples=False)
    model = SentenceTransformer(final_model_dir, model_kwargs=model_kwargs,model_card_data=model_card_data)
    logger.info(f"model.max_seq_length: {model.max_seq_length}")
    if model_shortname != "xlm-roberta-large":
        model.max_seq_length=32000
    logger.info(f"model.max_seq_length: {model.max_seq_length}")
    logger.info("Loading data")
    #task = mteb.get_task(task_name)
    task = mteb.get_task(task_name,languages=[args.lang],eval_splits=[args.split])

    logger.info("Evaluating")
    results = mteb.evaluate(model,task,cache=None,prediction_folder=model_predictions_folder,encode_kwargs=encode_kwargs)
    p = Path(task_results_path)
    results.to_disk(p)
    logger.info("Done")