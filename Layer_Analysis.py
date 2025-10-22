import json
import os
import copy
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


import gc

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from sklearn.metrics import roc_auc_score


#######################################################################################################################################################

def clean_gpu_memory(verbose=True):
    """Safely clear CUDA memory and Python garbage."""
    if verbose:
        print("Cleaning up GPU memory...")

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            if verbose:
                mem = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"GPU memory after cleanup: {mem:.2f} MB")
        elif torch.backends.mps.is_available():
            # For Apple Silicon (MPS)
            if verbose:
                print("MPS backend detected (Apple GPU). Memory is managed automatically.")
        else:
            if verbose:
                print("No GPU detected; nothing to clean.")
    except Exception as e:
        print(f"GPU cleanup failed: {e}")


def load_data(path):
    """
    Reads JSON and parses individual reasoning examples into a dictionary and finally returns samples.
    Returns:
    -items: list of dictionaries; [{'logic':<value>, 'topic':<value>, 'steps':[..]} , {....} , ...]
    """
    try:
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)
    except:
        raise Exception(f"Error in reading file from : {path} ") 
    items=[]
    for key,seqs in data.items():
        for row in seqs:
            try:
                lang=row.get("lang",None)
                if lang is not None:
                    steps=row.get("steps",[])
                    topic=row.get("topic","ABSTRACT")
                    items.append({"logic":key,"steps":steps,"topic":topic})
            except:
                continue
    return items


def _resample_to_len(arr, K):
    """Resample a 1D array to fixed length K via linear interpolation on [0,1]."""
    arr = np.asarray(arr, dtype=np.float32)
    if K <= 0:
        return np.zeros((0,), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((K,), dtype=np.float32)
    if arr.size == 1:
        return np.full((K,), float(arr[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=arr.size)
    x_new = np.linspace(0.0, 1.0, num=K)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def ensure_padding_tokens(tokenizer, model=None):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

#######################################################################################################################################################
#Helpers for extracting internal activations: "Vectors" for reasoning trajectories 

@torch.no_grad()
def _token_spans_for_steps(tokenizer, steps, accumulation="cumulative"):
    """
    Returns:
      - input_ids (1, T) for cumulative OR (B, T_max) for non-cumulative (padded)
      - attention_mask (same shape)
      - spans: list of (start, end) token indices per step
      - cumulative flag
    """
    if accumulation == "cumulative":
        # build one long context and compute spans per step, using an explicit newline separator
        spans = []
        pieces = []
        start = 0
        for t, s in enumerate(steps):
            piece = s if t == 0 else ("\n" + s)
            pieces.append(piece)
            ids = tokenizer(piece, add_special_tokens=False).input_ids
            end = start + len(ids)
            spans.append((start, end))
            start = end
        full_text = "".join(pieces)
        enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        return enc["input_ids"], attn, spans, True
    else:
        # tokenize each step separately, then pad/batch
        encs = [tokenizer(s, add_special_tokens=False).input_ids for s in steps]
        lens = [len(x) for x in encs]
        T = max(lens) if lens else 0
        B = len(encs)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        spans = []
        for i, ids in enumerate(encs):
            L = len(ids)
            if L:
                input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :L] = 1
            spans.append((0, L))  # each step is its whole sequence
        return input_ids, attention_mask, spans, False

@torch.no_grad()
def run_once_get_hidden_states(model, tokenizer, steps, accumulation="cumulative", device="cpu"):
    """
    To cache all hidden states of model after forward pass of one item (reasoning example/chain)
    Returns:
      - hs: [L, T, D] if cumulative else [L, B, T, D] (CPU, float32); hidden state of the model for a particular reasoning chain
      - spans: per-step (start, end)
      - cumulative: bool
    """
    model.eval()
    
    input_ids, attn_mask, spans, cumulative = _token_spans_for_steps(tokenizer, steps, accumulation)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    outs = model(input_ids=input_ids,
                 attention_mask=attn_mask,
                 output_hidden_states=True,
                 use_cache=False)

    # (embeddings, layer1, ..., layerN) -> stack -> [L_all, B, T, D]
    hs = torch.stack(outs.hidden_states, dim=0)
    if cumulative:
        hs = hs[:, 0]  # -> [L_all, T, D]

    # Move to CPU for cheaper pooling; keep float32
    hs = hs.to("cpu").float()
    return hs, spans, cumulative

def _safe_slice_1d(x, start, end):
    # returns a non-empty slice: if empty, backs off to the nearest valid single token
    if end > start:
        return x[start:end]
    T = x.shape[0]
    if T == 0:
        return x  # empty
    idx = min(max(end - 1, 0), T - 1)
    return x[idx:idx+1]

def _safe_slice_2d(x, i, start, end):
    if end > start:
        return x[i, start:end]
    T = x.shape[1]
    if T == 0:
        return x[i, 0:0]
    idx = min(max(end - 1, 0), T - 1)
    return x[i, idx:idx+1]

def pool_from_cache(hs, spans, cumulative, pooling="step_mean", context_k=16, channel=-1):
    """
    Extract hidden activation for a given layer (indexed by channel) : "vectors" for the reasoning trajectory
    hs: [L, T, D] if cumulative else [L, B, T, D]; hidden state of the model for a particular reasoning chain
    spans: list[(start, end)]
    chananel: index of hidden layer for which activations extracted
    
    returns: list[np.float32] (one vector per step)
    """
    Lall = hs.shape[0]
    if channel < -Lall or channel >= Lall:
        raise ValueError(f"channel index {channel} out of range for {Lall} layers")

    # same for cumulative/non-cumulative
    layer = hs[channel]  # [T, D] or [B, T, D]
    vecs = []

    if pooling == "context_mean":
        if cumulative:
            for _, end in spans:
                end = max(0, end)
                v = layer[:end].mean(dim=0) if end > 0 else layer.mean(dim=0) * 0
                vecs.append(v.numpy().astype(np.float32))
        else:
            for i, (_, end) in enumerate(spans):
                end = max(0, end)
                v = layer[i, :end].mean(dim=0) if end > 0 else layer[i, :0].mean(dim=0)
                v = v if end > 0 else torch.zeros(layer.shape[-1])
                vecs.append(v.numpy().astype(np.float32))

    elif pooling == "last":
        if cumulative:
            for _, end in spans:
                idx = max(0, end - 1)
                idx = min(idx, layer.shape[0]-1) if layer.shape[0] > 0 else 0
                v = layer[idx] if layer.shape[0] > 0 else torch.zeros(layer.shape[-1])
                vecs.append(v.numpy().astype(np.float32))
        else:
            for i, (_, end) in enumerate(spans):
                idx = max(0, end - 1)
                idx = min(idx, layer.shape[1]-1) if layer.shape[1] > 0 else 0
                v = layer[i, idx] if layer.shape[1] > 0 else torch.zeros(layer.shape[-1])
                vecs.append(v.numpy().astype(np.float32))

    else:
        # "step_mean" (default) or "context_aware_mean"
        step_mean = (pooling == "step_mean")
        k = max(0, int(context_k))

        if cumulative:
            prev_end = 0
            T = layer.shape[0]
            for (start, end) in spans:
                start = prev_end
                prev_end = end
                if step_mean:
                    sl = _safe_slice_1d(layer, start, end)
                else:
                    ctx_start = max(0, start - k)
                    sl = _safe_slice_1d(layer, ctx_start, end)
                v = sl.mean(dim=0) if sl.numel() else torch.zeros(layer.shape[-1])
                vecs.append(v.numpy().astype(np.float32))
        else:
            B, T = layer.shape[0], layer.shape[1]
            for i, (start, end) in enumerate(spans):
                if step_mean:
                    sl = _safe_slice_2d(layer, i, start, end)
                else:
                    ctx_start = max(0, start - k)
                    sl = _safe_slice_2d(layer, i, ctx_start, end)
                v = sl.mean(dim=0) if sl.numel() else torch.zeros(layer.shape[-1])
                vecs.append(v.numpy().astype(np.float32))

    return vecs


#######################################################################################################################################################
#Helper to extract velocity and curvature sequences from vectors of reasoning trajectories 

def compute_velocities_and_curvature(vectors, eps=1e-8, target_len=10):
    """
    vectors: dict[key] -> list of arrays; each array[i] has shape (T, D) storing y_1..y_T; key corresponds to individual logic-class
    target_len: resample each curvature sequence to this fixed length
    Returns:
    -velocities: dict[key] -> list of arrays; each array[i] is velocity associated with corresponding vector
    -curvature: dict[key] -> list of arrays; each array[i] is curvature (acceleration) associated with vector
    """
    velocities = {}
    curvature  = {}

    for key, seq_list in vectors.items():
        vel_list, curv_list, curv_fix_list = [], [], []

        for Y in seq_list:
            Y = np.asarray(Y, dtype=np.float32)  # (T, D)
            T = Y.shape[0]
            dY = np.diff(Y, axis=0)              # (T-1, D)
            vel_list.append(dY)

            if T >= 3:
                U = dY[:-1]                      # (T-2, D)
                V = dY[ 1:]                      # (T-2, D)
                Un = np.linalg.norm(U, axis=1) + eps
                Vn = np.linalg.norm(V, axis=1) + eps
                cos = (U * V).sum(1) / (Un * Vn)

                chord = np.linalg.norm(V + U, axis=1) + eps   #denominator:  ||y_{t+1} - y_{t-1}|| + epsilon
                kappa = 2.0 * np.sqrt(np.clip(1.0 - cos**2, 0.0, 1.0)) / chord

                tiny = (np.minimum(Un, Vn) < 1e-7) | (chord < 1e-7)
                kappa[tiny] = 0.0
            else:
                kappa = np.zeros((0,), dtype=np.float32)

            curv_list.append(_resample_to_len(kappa, target_len))


        velocities[key] = vel_list
        curvature[key]  = curv_list


    return (velocities, curvature)

#######################################################################################################################################################
#Helpers to calculate Pearson correlation for curvature sequences

def _stack_examples(data_dict, label_dict):
    seqs, names, labels = [], [], []
    for logic, lst in data_dict.items():
        L = len(lst)
        provided = label_dict.get(logic, [f"{logic}{i+1}" for i in range(L)])
        if len(provided) < L:
            provided = provided + [f"{logic}{i+1}" for i in range(len(provided), L)]
        elif len(provided) > L:
            provided = provided[:L]
        for i, seq in enumerate(lst):
            seqs.append(np.asarray(seq))
            names.append(str(provided[i]))
            labels.append(logic)
    return seqs, names, labels



def _pearson_1d(a, b, eps = 1e-8):
    """ Calculating Pearson correlation for two 1D sequences a & b """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    # constant checks
    a_const = np.allclose(a, a[0])
    b_const = np.allclose(b, b[0])
    if a_const and b_const:
        # same constant -> 1.0; different constants -> 0.0
        return 1.0 if np.allclose(a[0], b[0]) else 0.0
    if a_const or b_const:
        return 0.0  # undefined -> 0.0 
    # standard Pearson
    aa = (a - a.mean()) / (a.std() + eps)
    bb = (b - b.mean()) / (b.std() + eps)
    r = float((aa * bb).mean())
    # numerical guard
    if not np.isfinite(r):
        return 0.0
    return float(np.clip(r, -1.0, 1.0))

def get_pearson(data_dict,label_dict,K=13):
    """
    Returns an NxN symmetric Pearson matrix over resampled 1D sequences.
    """
    seqs, names, labels = _stack_examples(data_dict, label_dict)
    N = len(seqs)
    if N == 0:
        return np.zeros((0,0), dtype=np.float32)

    lengths = [len(np.asarray(s).reshape(-1)) for s in seqs]
    K_use = int(K) if K is not None else (min(lengths) if lengths else 0)

    proc = []
    for s in seqs:
        s1d = np.asarray(s, dtype=np.float32).reshape(-1)
        if s1d.size != K_use:
            s1d = _resample_to_len(s1d, K_use)
        proc.append(s1d)

    M = np.eye(N, dtype=np.float32)
    # fill upper triangle then mirror
    for i in range(N):
        for j in range(i+1, N):
            r = _pearson_1d(proc[i], proc[j])
            M[i, j] = r
            M[j, i] = r

    # final cleanup
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return M


#######################################################################################################################################################

def compute_logic_steering_metrics(
    sim_by_layer,
    labels,
    fisher_z= True,
    zscore_within_layer= False,
):
    """
    sim_by_layer: {layer_idx: (N,N) matrix} where entries are Pearson similarities of curvature. Must align with labels order.
    labels:       list of N strings like 'logicC:weather_en'. Logic class is taken as prefix before ':'.
    fisher_z:     apply Fisher atanh to stabilize and make layers comparable.
    zscore_within_layer: additionally z-score off-diagonals per layer (for optional visualization).

    Returns:
      {
        'layers': np.array of sorted layer indices,
        'mu_within': np.array[L],       # mean Fisher-z within-logic
        'mu_between': np.array[L],      # mean Fisher-z between-logic
        'delta': np.array[L],           # mu_within - mu_between
        'dprime': np.array[L],          # standardized separation
        'auc': np.array[L],             # ROC-AUC (within as positives)
        'n_within': np.array[L],
        'n_between': np.array[L],
        'per_layer': {layer: {'within_vals':..., 'between_vals':...}},
        'standardized_matrices': {layer: (N,N) np.ndarray}  # only if zscore_within_layer
      }
    """
    #0)basic checks
    layers = sorted(sim_by_layer.keys())
    if not layers:
        raise ValueError("sim_by_layer is empty.")
    N = None
    for L in layers:
        M = np.asarray(sim_by_layer[L])
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Layer {L}: matrix must be square; got shape {M.shape}.")
        if N is None:
            N = M.shape[0]
        elif M.shape[0] != N:
            raise ValueError(f"Layer {L}: matrix size {M.shape[0]} != {N} (labels length mismatch?).")
    if len(labels) != N:
        raise ValueError(f"labels length {len(labels)} != matrix size {N}.")

    #1) parse logic classes
    #logic = prefix before ':' ; fallback to full string if ':' absent
    logic = [s.split(':', 1)[0] if isinstance(s, str) else str(s) for s in labels]
    logic = np.asarray(logic)

    #2) index masks (upper triangle, no diagonal)
    iu = np.triu_indices(N, k=1)

    #Within-logic and between-logic masks over (i,j) pairs
    within_mask_pairs = (logic[iu[0]] == logic[iu[1]])
    between_mask_pairs = ~within_mask_pairs

    #3) storage
    Lm = len(layers)
    mu_w   = np.full(Lm, np.nan, dtype=np.float64)
    mu_b   = np.full(Lm, np.nan, dtype=np.float64)
    delta  = np.full(Lm, np.nan, dtype=np.float64)
    dprime = np.full(Lm, np.nan, dtype=np.float64)
    auc    = np.full(Lm, np.nan, dtype=np.float64)
    n_w    = np.zeros(Lm, dtype=int)
    n_b    = np.zeros(Lm, dtype=int)
    per_layer = {}
    standardized_mats = {} if zscore_within_layer else None

    #4) per-layer processing
    for idx, L in enumerate(layers):
        M = np.asarray(sim_by_layer[L], dtype=np.float64)

        #Fisher z on all entries; leave diagonal unused later
        if fisher_z:
            # clip to avoid inf at exactly ±1
            M = np.clip(M, -0.999999, 0.999999)
            Mz = np.arctanh(M)
        else:
            Mz = M

        #Collect upper-triangle off-diagonals
        vals = Mz[iu]
        vals = vals.astype(np.float64)

        w_vals = vals[within_mask_pairs]
        b_vals = vals[between_mask_pairs]
        n_w[idx] = w_vals.size
        n_b[idx] = b_vals.size

        if w_vals.size == 0 or b_vals.size == 0:
            per_layer[L] = {'within_vals': w_vals, 'between_vals': b_vals}
            continue

        #Means and effect sizes
        mu_w[idx] = float(np.mean(w_vals))
        mu_b[idx] = float(np.mean(b_vals))
        delta[idx] = mu_w[idx] - mu_b[idx]

        #pooled SD for d'
        sw2 = float(np.var(w_vals, ddof=1)) if w_vals.size > 1 else 0.0
        sb2 = float(np.var(b_vals, ddof=1)) if b_vals.size > 1 else 0.0
        denom = np.sqrt(0.5 * (sw2 + sb2)) + 1e-12
        dprime[idx] = delta[idx] / denom if denom > 0 else np.nan

        #AUC: within as positives, between as negatives
        y_true = np.concatenate([np.ones_like(w_vals), np.zeros_like(b_vals)])
        y_score = np.concatenate([w_vals, b_vals])
        try:
            #If y_score constant, AUC is undefined;
            if np.allclose(y_score, y_score[0]):
                auc[idx] = np.nan
            else:
                auc[idx] = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc[idx] = np.nan

        per_layer[L] = {'within_vals': w_vals, 'between_vals': b_vals}

        #within-layer z-scored matrix (for visualization/consistency)
        if zscore_within_layer:
            #z-score only off-diagonals, keep diag as 0
            off = vals  # this is already Mz on upper triangle
            mu = float(off.mean())
            sd = float(off.std(ddof=1)) if off.size > 1 else 0.0
            Z = Mz.copy()
            if sd > 0:
                # z-score symmetrically (upper & lower)
                Zu = (Mz[iu] - mu) / sd
                Z[iu] = Zu
                Z[(iu[1], iu[0])] = Zu  # mirror
            else:
                Z = np.zeros_like(Mz)
            np.fill_diagonal(Z, 0.0)
            standardized_mats[L] = Z

    out = {
        'layers': np.array(layers, dtype=int),
        'mu_within': mu_w,
        'mu_between': mu_b,
        'delta': delta,
        'dprime': dprime,
        'auc': auc,
        'n_within': n_w,
        'n_between': n_b,
        'per_layer': per_layer,
    }
    if standardized_mats is not None:
        out['standardized_matrices'] = standardized_mats
    return out


#######################################################################################################################################################

def main():
    ap = argparse.ArgumentParser(description="Layer by Layer evolution of curvature for reasoning trajectories using HF models")
    ap.add_argument("--hf_model", type=str, required=True, help="HuggingFace model ID/path")
    ap.add_argument("--data_path", type=str, default="./data/all_final_data.json")
    
    ap.add_argument("--pooling", type=str, default="step_mean", choices=["step_mean", "context_mean", "last", "context_aware_mean"])
    ap.add_argument("--accumulation", type=str, default="cumulative", choices=["cumulative", "isolated"])
    ap.add_argument("--context_aware_k", type=int, default=16)
    
    ap.add_argument("--resample_len", type=int, default=13, help="Resample curvature sequences to target length") 
    ap.add_argument("--max_samples_per_class", type=int, default=60, help="Max samples for each logic-class") 
    
    ap.add_argument("--start_layer", type=int, default=0, help="Index of initial layer to analyse; default: 0 => for starting with embedding layer")
    ap.add_argument("--end_layer", type=int, default=28, help="Index of final layer to analyse")
    
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="./exp")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    items=load_data(args.data_path)

    model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(args.device)
    tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    ensure_padding_tokens(tok, model)
    
    K=args.resample_len #resample length
    considered_logics=['logicA','logicB','logicC','logicD','logicE']#Choose specific logic-classes 

    ALL_VECTORS = {}
    ALL_CURVATURES={}
    ALL_VELOCITIES={}
    ALL_PEARSONS={}

    LABELS_BY_LOGIC = defaultdict(list)
    CACHE_BY_LOGIC = defaultdict(list)   # logic -> list of (hs, spans, cumulative)

    #One forward pass per item; cache on CPU
    print(">>> CACHING HIDDEN STATES")
    for item in items:
        logic = item['logic']
        if logic not in considered_logics:
            continue
        if len(LABELS_BY_LOGIC[logic]) >= args.max_samples_per_class:
            continue

        # forward pass (once)
        hs, spans, cumulative = run_once_get_hidden_states(
            model, tok, item['steps'],
            accumulation=args.accumulation,
            device=args.device
        )
        # hs is already moved to CPU inside run_once_get_hidden_states()
        CACHE_BY_LOGIC[logic].append((hs, spans, cumulative))
        LABELS_BY_LOGIC[logic].append(f"{logic}:{item['topic']}")
        

    #free GPU memory as we moved hs to CPU
    clean_gpu_memory()
    
    print(">>>EXTRACTING VECTORS FOR CHANNELS (Layers)")
    #For each channel, pool from cached hidden states
    for ch_hs in range(args.start_layer, args.end_layer + 1):
        #print(">>>>> CHANNEL", ch_hs)
        vectors = {logic: [] for logic in CACHE_BY_LOGIC.keys()}
        
        for logic, entries in CACHE_BY_LOGIC.items():
            for hs, spans, cumulative in entries:
                vecs = pool_from_cache(
                    hs, spans, cumulative,
                    pooling=args.pooling,
                    channel=ch_hs,
                    context_k=args.context_aware_k
                )
                # vecs: list[np.float32] (one vector per step)
                vectors[logic].append(vecs)

        # store per-channel results
        ALL_VECTORS[ch_hs] = (vectors, dict(LABELS_BY_LOGIC))
        
    LABELS=[]
    for k in LABELS_BY_LOGIC.keys():
        LABELS+=LABELS_BY_LOGIC[k]
    
    del CACHE_BY_LOGIC,LABELS_BY_LOGIC
    
    print(">>>EXTRACTING HIGHER-ORDER CHARACTERISTICS FOR CHANNELS (Layers)")
    for key in ALL_VECTORS.keys():
        #print(">>>>CHANNEL : ",key)
        velocities,curvature=compute_velocities_and_curvature(ALL_VECTORS[key][0], target_len=K)
        ALL_VELOCITIES[key]= copy.deepcopy(velocities)
        ALL_CURVATURES[key]= copy.deepcopy(curvature)

        M=get_pearson(ALL_CURVATURES[key],ALL_VECTORS[key][1], K=K)
        ALL_PEARSONS[key]= copy.deepcopy(M)

    OUTS=compute_logic_steering_metrics(ALL_PEARSONS,LABELS,True,True)
    
    try:
        np.save(f"{args.save_dir}/ALL_VECTORS.npy", ALL_VECTORS)
        np.save(f"{args.save_dir}/ALL_VELOCITIES.npy", ALL_VELOCITIES)
        np.save(f"{args.save_dir}/ALL_CURVATURES.npy", ALL_CURVATURES)
        np.save(f"{args.save_dir}/ALL_PEARSONS.npy",ALL_PEARSONS)
        np.save(f"{args.save_dir}/OUTS.npy",OUTS)
    except Exception as e:
        print(f">> ERROR IN SAVING : {str(e)}")

    plt.plot(OUTS['layers'],OUTS['delta'])
    plt.xlabel("Layer_index")
    plt.ylabel(r"$\delta$")
    plt.grid(which="major")
    plt.savefig(f"{args.save_dir}/delta.png")
    plt.show()
    
    plt.plot(OUTS['layers'],OUTS['dprime'])
    plt.xlabel("Layer_index")
    plt.ylabel(r"$\delta$")
    plt.grid(which="major")
    plt.savefig(f"{args.save_dir}/deltaprime.png")
    plt.show()
    
    plt.plot(OUTS['layers'],OUTS['auc'])
    plt.xlabel("Layer_index")
    plt.ylabel(r"auc")
    plt.grid(which="major")
    plt.savefig(f"{args.save_dir}/auc.png")
    plt.show()
    
    

if __name__ == "__main__":
    main()