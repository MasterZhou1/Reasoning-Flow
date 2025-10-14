#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils import split_cot_steps
from utils_stat import pairwise_similarity, pairwise_menger_curvature_similarity, plot_similarity_heatmap
from utils import plot_trajectories_pca


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class LogicItem:
    logic: str  # allow any logic key (e.g., LogicA, LogicB, LogicC, ...)
    topic: Optional[str]
    steps: List[str]


def load_dataset_any_logic(path: str) -> List[LogicItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[LogicItem] = []
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
            topic_str: Optional[str] = str(topic_val) if topic_val is not None else None
            items.append(LogicItem(logic=str(logic_key), topic=topic_str, steps=steps))
    return items


# ------------------------------
# Step representation via token hidden states
# ------------------------------
@torch.no_grad()
def step_vectors_for_sequence(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    steps: List[str],
    *,
    pooling: Literal["step_mean", "context_mean", "last", "context_aware_mean"] = "step_mean",
    accumulation: Literal["cumulative", "isolated"] = "cumulative",
    context_aware_k: int = 16,
    device: str = "cpu",
) -> List[np.ndarray]:
    vecs: List[np.ndarray] = []
    prev_input_ids: torch.Tensor | None = None
    context = ""

    for t, step in enumerate(steps):
        if accumulation == "cumulative":
            context = step if t == 0 else (context + "\n" + step)
        else:
            context = step

        # Use add_special_tokens=False to keep alignment consistent across steps
        enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        # Prefer last hidden state from outputs.last_hidden_state when available
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hs = outputs.last_hidden_state  # [1, L, D]
        else:
            hs = outputs.hidden_states[-1]  # [1, L, D]

        L = input_ids.shape[1]
        D = hs.shape[-1]

        if pooling == "context_mean":
            v = hs.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
        elif pooling == "last":
            v = hs[:, -1, :].squeeze(0).detach().float().cpu().numpy()
        else:
            # Pool over tokens of the current step (new tokens since previous context)
            if accumulation == "cumulative" and prev_input_ids is not None:
                prev_len = prev_input_ids.shape[1]
            else:
                prev_len = 0
            start = min(prev_len, L)  # guard
            step_slice = hs[:, start:, :] if start < L else hs[:, -1:, :]
            if pooling == "step_mean":
                v = step_slice.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
            else:
                # context_aware_mean: include K tokens before the step span
                k = max(0, int(context_aware_k))
                ctx_start = max(0, start - k)
                ctx_slice = hs[:, ctx_start:, :]
                v = ctx_slice.mean(dim=1).squeeze(0).detach().float().cpu().numpy()

        vecs.append(v.astype(np.float32))
        prev_input_ids = input_ids

    return vecs


def build_label(item: LogicItem) -> str:
    if item.topic is None or str(item.topic).strip() == "":
        return f"{item.logic}:abstract"
    return f"{item.logic}:{item.topic}"


def group_by_logic(items: List[LogicItem]) -> Dict[str, List[LogicItem]]:
    groups: Dict[str, List[LogicItem]] = {}
    for it in items:
        groups.setdefault(it.logic, []).append(it)
    return groups


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_label(label: str) -> str:
    """Sanitize label for filesystem paths."""
    s = label.strip().lower()
    for ch in ["/", "\\", ":", " ", ",", "|", "*", "?", "\n", "\t", "(", ")", "[", "]"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._")


def main():
    ap = argparse.ArgumentParser(description="Hidden state analysis for HF models: logic vs semantics")
    ap.add_argument("--hf_model", type=str, required=False, help="HF model path or ID (comma-separated supported via --hf_models)")
    ap.add_argument("--hf_models", type=str, default=None, help="Comma-separated list of HF model IDs/paths; when set, runs analysis for each model separately")
    ap.add_argument("--data_file", type=str, default="data/generated_logic_topics.json", help="JSON dataset file")
    ap.add_argument("--pooling", type=str, default="step_mean", choices=["step_mean", "context_mean", "last", "context_aware_mean"], help="Pooling strategy for step vector")
    ap.add_argument("--accumulation", type=str, default="cumulative", choices=["cumulative", "isolated"], help="Context accumulation strategy")
    ap.add_argument("--context_aware_k", type=int, default=16, help="K tokens of context to include for context_aware_mean")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # Large model support
    ap.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    ap.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    ap.add_argument("--device_map", type=str, default="auto", help="Accelerate device map for 4/8-bit (e.g., auto, balanced, sequential)")
    ap.add_argument("--dtype", type=str, default=None, choices=["auto", "fp16", "fp32", "bf16"], help="Override torch dtype for model weights (ignored for 4/8-bit)")
    ap.add_argument("--attn_implementation", type=str, default=None, help="Attention backend (e.g., flash_attention_2, sdpa)")
    ap.add_argument("--sections", type=str, default="all", help="Comma-separated logic keys to include (e.g., LogicA,LogicB). Use 'all' for all keys in data file.")
    ap.add_argument("--similarity_order", type=int, default=1, help="Similarity type selector: 0=positions, 1=Δ, 2=Δ², 3=Menger curvature")
    ap.add_argument("--save_dir", type=str, default="figs")
    ap.add_argument("--hide_axis_text", action="store_true", help="If set, hide x/y tick labels on similarity heatmaps")
    ap.add_argument("--color_scale", type=str, default="RdBu_r", help="Plotly color scale for heatmap (e.g., 'Blues', 'Greys', 'Viridis', 'RdBu_r')")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    # Optional: parse JSON list for custom color scale
    try:
        if isinstance(args.color_scale, str) and args.color_scale.strip().startswith("["):
            import json as _json
            args.color_scale = _json.loads(args.color_scale)
    except Exception:
        pass

    # Validate model specification
    if not args.hf_model and not args.hf_models:
        raise SystemExit("Provide --hf_model or --hf_models (comma-separated).")

    # Load dataset
    items = load_dataset_any_logic(args.data_file)
    if args.sections != "all":
        keep = {s.strip() for s in args.sections.split(",") if s.strip()}
        items = [it for it in items if it.logic in keep]
    if len(items) == 0:
        raise SystemExit("No items loaded. Check data_file or sections filter.")

    # Resolve models (single or multiple)
    model_ids: List[str]
    if args.hf_models:
        model_ids = [m.strip() for m in args.hf_models.split(",") if m.strip()]
    else:
        model_ids = [args.hf_model]

    def _select_dtype() -> torch.dtype:
        if args.dtype in (None, "auto"):
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if args.dtype == "fp16":
            return torch.float16
        if args.dtype == "fp32":
            return torch.float32
        if args.dtype == "bf16":
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def _safe_load_model(mid: str):
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        # Provide pad token for decoder-only models (e.g., LLaMA) when missing
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        try:
            tok.padding_side = "left"
        except Exception:
            pass

        torch_dtype = _select_dtype()
        common_kwargs = {"trust_remote_code": True}
        if args.attn_implementation:
            common_kwargs["attn_implementation"] = args.attn_implementation

        model = None
        # Try quantized load if requested
        if args.load_in_4bit or args.load_in_8bit:
            qkwargs = dict(common_kwargs)
            qkwargs["device_map"] = args.device_map
            if args.load_in_4bit:
                qkwargs["load_in_4bit"] = True
            if args.load_in_8bit:
                qkwargs["load_in_8bit"] = True
            try:
                model = AutoModel.from_pretrained(mid, **qkwargs)
            except Exception:
                try:
                    model = AutoModelForCausalLM.from_pretrained(mid, **qkwargs)
                except Exception as e:
                    print(f"[WARN] 4/8-bit load failed for {mid}: {e}. Falling back to full-precision.")

        if model is None:
            rkwargs = dict(common_kwargs)
            rkwargs["torch_dtype"] = torch_dtype
            try:
                model = AutoModel.from_pretrained(mid, **rkwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(mid, **rkwargs)
            model.to(args.device)

        model.eval()
        return tok, model

    # Run analysis for each model id
    for mid in model_ids:
        print(f"[INFO] Loading model: {mid}")
        tokenizer, model = _safe_load_model(mid)

        # Build step vectors per item
        label2steps: Dict[str, List[np.ndarray]] = {}
        label2meta: Dict[str, Dict[str, Optional[str] | int]] = {}
        for it in items:
            vecs = step_vectors_for_sequence(
                tokenizer, model, it.steps,
                pooling=args.pooling,
                accumulation=args.accumulation,
                context_aware_k=args.context_aware_k,
                device=args.device,
            )
            label = build_label(it)
            label2steps[label] = vecs
            label2meta[label] = {"logic": it.logic, "topic": it.topic, "num_steps": len(it.steps)}

        # Choose save root (per-model subfolder if multiple)
        model_tag = mid.split("/")[-1].strip() or _safe_label(mid)
        save_root = args.save_dir if len(model_ids) == 1 else os.path.join(args.save_dir, _safe_label(model_tag))
        ensure_dir(save_root)

        # Save raw step vectors and steps text for analysis
        data_root = os.path.join(save_root, "data")
        emb_dir = os.path.join(data_root, "embeddings")
        steps_dir = os.path.join(data_root, "steps")
        ensure_dir(emb_dir)
        ensure_dir(steps_dir)

        manifest = {
            "pooling": args.pooling,
            "accumulation": args.accumulation,
            "context_aware_k": int(args.context_aware_k),
            "similarity_order": int(args.similarity_order),
            "sections": args.sections,
            "hf_model": mid,
            "device": args.device,
            "labels": list(label2steps.keys()),
            "load_in_8bit": bool(args.load_in_8bit),
            "load_in_4bit": bool(args.load_in_4bit),
            "device_map": args.device_map,
            "dtype": args.dtype or "auto",
            "attn_implementation": args.attn_implementation or None,
        }

        for it in items:
            label = build_label(it)
            safe = _safe_label(label)
            vecs = label2steps[label]
            arr = np.stack(vecs).astype(np.float32)
            np.save(os.path.join(emb_dir, f"{safe}.npy"), arr)
            with open(os.path.join(steps_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
                json.dump({"label": label, "logic": it.logic, "topic": it.topic, "steps": it.steps}, f, ensure_ascii=False, indent=2)

        with open(os.path.join(data_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # Plot PCA trajectories per logic group (one per logic)
        groups = group_by_logic(items)
        for logic_key, group_items in groups.items():
            if not group_items:
                continue
            sub = {build_label(it): label2steps[build_label(it)] for it in group_items}

            # Save PCA 3D projections for analysis (same PCA as used in the plot)
            # Fit PCA on all points in this logic group
            all_points = np.vstack([vec for embs in sub.values() for vec in embs])
            from sklearn.decomposition import PCA as _PCA
            pca_model = _PCA(n_components=3, random_state=42).fit(all_points)
            pca_dir = os.path.join(data_root, "pca", logic_key)
            ensure_dir(pca_dir)
            # Save PCA model components
            np.savez_compressed(
                os.path.join(pca_dir, "pca_model.npz"),
                components_=pca_model.components_.astype(np.float32),
                explained_variance_=pca_model.explained_variance_.astype(np.float32),
                mean_=pca_model.mean_.astype(np.float32),
            )
            # Save per-label projected trajectories as CSV
            for label, embs in sub.items():
                traj = np.vstack(embs)
                proj = pca_model.transform(traj)
                df_proj = pd.DataFrame({
                    "t": np.arange(1, proj.shape[0] + 1, dtype=int),
                    "pc1": proj[:, 0].astype(np.float32),
                    "pc2": proj[:, 1].astype(np.float32),
                    "pc3": proj[:, 2].astype(np.float32),
                })
                df_proj.to_csv(os.path.join(pca_dir, f"{_safe_label(label)}_pca3d.csv"), index=False)

            plot_trajectories_pca(
                sub,
                exclude_prompt=False,
                title=f"{logic_key} Trajectories (PCA) [{args.pooling}/{args.accumulation}] — {model_tag}",
                save_pdf_path=os.path.join(save_root, f"{logic_key}_trajectories_pca.pdf"),
                width=1000,
                height=700,
            )

        # Single global similarity heatmap across all logic and topics
        order_sel = int(args.similarity_order)
        if order_sel == 3:
            labels_all, sim_all = pairwise_menger_curvature_similarity(label2steps, metric="pearson", align="truncate")
            title = f"Global Similarity (Menger curvature) — {model_tag}"
            base_name = "global_similarity_order3_menger"
        else:
            labels_all, sim_all = pairwise_similarity(label2steps, order=order_sel, metric="mean_cos")
            title = f"Global Similarity (order={order_sel}) — {model_tag}"
            base_name = f"global_similarity_order{order_sel}"

        plot_similarity_heatmap(
            sim_all,
            labels=labels_all,
            title=title,
            width=1100,
            height=1000,
            save_pdf_path=os.path.join(save_root, f"{base_name}.pdf"),
            show_axis_text=not args.hide_axis_text,
            color_scale=args.color_scale,
        )

        # Save global similarity matrix for analysis
        df_sim = pd.DataFrame(sim_all, index=labels_all, columns=labels_all)
        df_sim.to_csv(os.path.join(data_root, f"{base_name}.csv"))
        np.save(os.path.join(data_root, f"{base_name}.npy"), sim_all.astype(np.float32))

        print("Analysis complete. Figures saved to:", os.path.abspath(save_root))
        print("Data saved to:", os.path.abspath(data_root))


if __name__ == "__main__":
    main()
