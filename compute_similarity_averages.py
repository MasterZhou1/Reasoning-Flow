#!/usr/bin/env python3
"""
Compute group-averaged similarity scores on a dataset of logic/topic sequences
using hidden-state step embeddings from HF models, similar to cot-hidden-dynamic.py.

For each specified similarity order, report macro/micro averages for:
- within-logic: pairs among sequences of the same logic
- within-topic: pairs among sequences of the same base topic (language suffix removed)
- within-language: pairs among sequences sharing the same language code

Usage example:
  python compute_similarity_averages.py \
    --hf_models /home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B,\
/home/users/your_name/store/pretrain/Qwen/Qwen3-1.7B,\
/home/users/your_name/store/pretrain/Qwen/Qwen3-4B,\
/home/users/your_name/store/pretrain/Qwen/Qwen3-8B,\
/home/users/your_name/store/pretrain/meta-llama/Meta-Llama-3-8B-Instruct \
    --data_file data/all_final_data.json \
    --orders 0,1,2,3 \
    --pooling step_mean --accumulation cumulative \
    --save_dir results/averages

Outputs per model under save_dir: a JSON and CSV with aggregated means.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils import split_cot_steps
from utils_stat import pairwise_similarity, pairwise_menger_curvature_similarity


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class Item:
    logic: str
    topic: Optional[str]
    lang: Optional[str]
    steps: List[str]


def load_dataset(path: str) -> List[Item]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[Item] = []
    for logic_key, seq_list in data.items():
        if not isinstance(seq_list, list):
            continue
        for rec in seq_list:
            if not isinstance(rec, dict) or "steps" not in rec:
                continue
            steps = rec.get("steps", [])
            if isinstance(steps, str):
                steps = split_cot_steps(steps)
            elif isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], str):
                steps = split_cot_steps(steps[0])
            topic_val = rec.get("topic", None)
            lang_val = rec.get("lang", None)
            topic_str: Optional[str] = str(topic_val) if topic_val is not None else None
            lang_str: Optional[str] = str(lang_val) if lang_val is not None else None
            items.append(Item(logic=str(logic_key), topic=topic_str, lang=lang_str, steps=steps))
    return items


def build_label(it: Item) -> str:
    if it.topic is None or str(it.topic).strip() == "":
        return f"{it.logic}:abstract"
    return f"{it.logic}:{it.topic}"


# ------------------------------
# HF model loading and step vectors
# ------------------------------
@torch.no_grad()
def step_vectors_for_sequence(
    tokenizer,
    model,
    steps: List[str],
    *,
    pooling: str = "step_mean",
    accumulation: str = "cumulative",
    context_aware_k: int = 16,
    device: str = "cpu",
) -> List[np.ndarray]:
    vecs: List[np.ndarray] = []
    prev_input_ids = None
    context = ""

    for t, step in enumerate(steps):
        if accumulation == "cumulative":
            context = step if t == 0 else (context + "\n" + step)
        else:
            context = step

        enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hs = outputs.last_hidden_state
        else:
            hs = outputs.hidden_states[-1]

        L = input_ids.shape[1]

        if pooling == "context_mean":
            v = hs.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
        elif pooling == "last":
            v = hs[:, -1, :].squeeze(0).detach().float().cpu().numpy()
        else:
            if accumulation == "cumulative" and prev_input_ids is not None:
                prev_len = prev_input_ids.shape[1]
            else:
                prev_len = 0
            start = min(prev_len, L)
            step_slice = hs[:, start:, :] if start < L else hs[:, -1:, :]
            if pooling == "step_mean":
                v = step_slice.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
            else:
                k = max(0, int(context_aware_k))
                ctx_start = max(0, start - k)
                ctx_slice = hs[:, ctx_start:, :]
                v = ctx_slice.mean(dim=1).squeeze(0).detach().float().cpu().numpy()

        vecs.append(v.astype(np.float32))
        prev_input_ids = input_ids

    return vecs


def _safe_label(s: str) -> str:
    ss = s.strip().lower()
    for ch in ["/", "\\", ":", " ", ",", "|", "*", "?", "\n", "\t", "(", ")", "[", "]"]:
        ss = ss.replace(ch, "_")
    while "__" in ss:
        ss = ss.replace("__", "_")
    return ss.strip("._")


def load_hf_model_quant(mid: str, *, device: str = "cuda:0", load_in_4bit: bool = False, load_in_8bit: bool = False, device_map: str = "auto"):
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass

    model = None
    if load_in_4bit or load_in_8bit:
        qkwargs = {"trust_remote_code": True, "device_map": device_map}
        if load_in_4bit:
            qkwargs["load_in_4bit"] = True
        if load_in_8bit:
            qkwargs["load_in_8bit"] = True
        try:
            model = AutoModel.from_pretrained(mid, **qkwargs)
        except Exception:
            try:
                model = AutoModelForCausalLM.from_pretrained(mid, **qkwargs)
            except Exception:
                model = None

    if model is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        common_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        try:
            model = AutoModel.from_pretrained(mid, **common_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(mid, **common_kwargs)
        model.to(device)

    model.eval()
    return tok, model


# ------------------------------
# Group utilities
# ------------------------------
LANG_CODES = {"en", "zh", "de", "ja"}


def base_topic(topic: Optional[str], lang: Optional[str]) -> Optional[str]:
    if topic is None:
        return None
    if lang:
        suf = f"_{lang}"
        if topic.endswith(suf):
            return topic[: -len(suf)]
    # best-effort fallback by suffix pattern
    parts = topic.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in LANG_CODES:
        return parts[0]
    return topic


def build_groups(labels: List[str], meta: Dict[str, dict]):
    by_logic: Dict[str, List[str]] = {}
    by_topic: Dict[str, List[str]] = {}
    by_lang: Dict[str, List[str]] = {}
    for lbl in labels:
        m = meta[lbl]
        logic = str(m.get("logic"))
        topic = m.get("topic")
        lang = m.get("lang")
        by_logic.setdefault(logic, []).append(lbl)
        bt = base_topic(topic, lang)
        if bt not in (None, "", "abstract"):
            by_topic.setdefault(str(bt), []).append(lbl)
        if lang not in (None, ""):
            by_lang.setdefault(str(lang), []).append(lbl)
    return by_logic, by_topic, by_lang


def mean_of_group_pairs(sim_labels: List[str], sim_mat: np.ndarray, groups: Dict[str, List[str]]) -> Tuple[float, float, Dict[str, float]]:
    """
    Return (macro_mean, micro_mean, per_group_means).
    - macro_mean: simple average of per-group means
    - micro_mean: pair-weighted average across all within-group pairs
    """
    idx = {lbl: i for i, lbl in enumerate(sim_labels)}
    per_group: Dict[str, float] = {}
    macro_vals: List[float] = []
    num_all_pairs = 0
    sum_all_pairs = 0.0
    for key, members in groups.items():
        ids = [idx[m] for m in members if m in idx]
        if len(ids) < 2:
            continue
        # upper triangle pairs
        s = 0.0
        c = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s += float(sim_mat[ids[i], ids[j]])
                c += 1
        if c == 0:
            continue
        mu = s / c
        per_group[key] = mu
        macro_vals.append(mu)
        num_all_pairs += c
        sum_all_pairs += s
    macro = float(np.mean(macro_vals)) if macro_vals else float("nan")
    micro = (sum_all_pairs / num_all_pairs) if num_all_pairs > 0 else float("nan")
    return macro, micro, per_group


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Group-averaged similarity over logic/topic/lang using HF models")
    ap.add_argument("--hf_models", type=str, required=True, help="Comma-separated HuggingFace model IDs/paths")
    ap.add_argument("--data_file", type=str, default="data/all_final_data.json")
    ap.add_argument("--orders", type=str, default="0,1,2,3", help="Comma-separated similarity orders: 0,1,2,3 (3=Menger curvature)")
    ap.add_argument("--pooling", type=str, default="step_mean", choices=["step_mean", "context_mean", "last", "context_aware_mean"])
    ap.add_argument("--accumulation", type=str, default="cumulative", choices=["cumulative", "isolated"])
    ap.add_argument("--context_aware_k", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    ap.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    ap.add_argument("--device_map", type=str, default="auto", help="Accelerate device map for 4/8-bit (e.g., auto, balanced, sequential)")
    ap.add_argument("--save_dir", type=str, default="results/averages")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    items = load_dataset(args.data_file)
    if len(items) == 0:
        raise SystemExit("No items loaded from data_file.")

    model_ids = [m.strip() for m in args.hf_models.split(",") if m.strip()]
    orders = [int(x) for x in args.orders.split(",") if x.strip()]
    import tqdm
    for mid in model_ids:
        print(f"[INFO] Loading model: {mid}")
        tokenizer, model = load_hf_model_quant(mid, device=args.device, load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit, device_map=args.device_map)

        # Build step vectors and metadata
        label2steps: Dict[str, List[np.ndarray]] = {}
        meta: Dict[str, dict] = {}

        for it in tqdm.tqdm(items):
            vecs = step_vectors_for_sequence(
                tokenizer, model, it.steps,
                pooling=args.pooling,
                accumulation=args.accumulation,
                context_aware_k=args.context_aware_k,
                device=args.device,
            )
            label = build_label(it)
            label2steps[label] = vecs
            meta[label] = {"logic": it.logic, "topic": it.topic, "lang": it.lang, "num_steps": len(it.steps)}

        labels = list(label2steps.keys())
        by_logic, by_topic, by_lang = build_groups(labels, meta)

        # Compute similarities per order and aggregate
        results = {"model": mid, "pooling": args.pooling, "accumulation": args.accumulation, "orders": {}}
        for ord_val in orders:
            print(f"[INFO] Computing similarities (order={ord_val}) ...")
            if ord_val == 3:
                sim_labels, sim_mat = pairwise_menger_curvature_similarity(label2steps, metric="pearson", align="truncate")
            else:
                sim_labels, sim_mat = pairwise_similarity(label2steps, order=ord_val, metric="mean_cos", align="truncate")

            # Group averages
            logic_macro, logic_micro, _ = mean_of_group_pairs(sim_labels, sim_mat, by_logic)
            topic_macro, topic_micro, _ = mean_of_group_pairs(sim_labels, sim_mat, by_topic)
            lang_macro, lang_micro, _ = mean_of_group_pairs(sim_labels, sim_mat, by_lang)

            results["orders"][str(ord_val)] = {
                "logic_macro_mean": logic_macro,
                "logic_micro_mean": logic_micro,
                "topic_macro_mean": topic_macro,
                "topic_micro_mean": topic_micro,
                "lang_macro_mean": lang_macro,
                "lang_micro_mean": lang_micro,
                "num_logics": len(by_logic),
                "num_topics": len(by_topic),
                "num_langs": len(by_lang),
                "num_labels": len(sim_labels),
            }

        # Save per-model JSON
        tag = mid.split("/")[-1] or _safe_label(mid)
        out_json = os.path.join(args.save_dir, f"averages_{_safe_label(tag)}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Also write a compact CSV-like TSV for quick viewing
        out_tsv = os.path.join(args.save_dir, f"averages_{_safe_label(tag)}.tsv")
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("order\tlogic_macro\tlogic_micro\ttopic_macro\ttopic_micro\tlang_macro\tlang_micro\n")
            for ord_val in orders:
                r = results["orders"][str(ord_val)]
                f.write(
                    f"{ord_val}\t{r['logic_macro_mean']:.6f}\t{r['logic_micro_mean']:.6f}\t"
                    f"{r['topic_macro_mean']:.6f}\t{r['topic_micro_mean']:.6f}\t"
                    f"{r['lang_macro_mean']:.6f}\t{r['lang_micro_mean']:.6f}\n"
                )
        print(f"[OK] Saved: {out_json} and {out_tsv}")


if __name__ == "__main__":
    main()
