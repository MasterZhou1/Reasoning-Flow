#!/usr/bin/env python3
"""
Generate a dataset similar to data/cot_data_logic.json using prompt templates
in the prompt/ folder and a text-generation backend (OpenAI or local HF).

Examples:
  # OpenAI backend (set OPENAI_API_KEY in env)
  python generate_dataset.py \
    --backend openai --model gpt-4o-mini \
    --prompt_template prompt/prompt_natural.md \
    --seeds_file data/seeds.jsonl \
    --out data/generated_cot_data_logic.json

  # Local HF backend (point to an instruction-tuned model path or id)
  python generate_dataset.py \
    --backend hf --hf_model /path/to/instruct-model \
    --prompt_template prompt/prompt2.md \
    --seeds_file data/seeds.jsonl \
    --out data/generated_cot_data_logic.json

Seeds file format (JSONL; one JSON object per line):
  {
    "section": "LogicA",              # required: dataset group key
    "topic": "weather",               # optional
    "which_prompt": "prompt2",        # one of {prompt, prompt2, prompt_natural}
    "N": 10,                           # required: number of steps/sentences
    "input": "如果今天下雨...",        # required for prompt2/prompt_natural
    "goal_formula": "(P→Q)∧P→Q"       # optional; used by 'prompt'
  }

Output JSON matches cot_data_logic.json structure: {section: [ {topic?, steps: [...]}, ... ]}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List


# --------- Backends ---------
def _gen_openai(prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 1200, *, api_key: Optional[str] = None) -> str:
    from openai import OpenAI  # requires openai>=1.0
    import os as _os
    _key = api_key or _os.environ.get("OPENAI_API_KEY")
    if not _key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY env or pass --openai_api_key.")
    client = OpenAI(api_key=_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # temperature=temperature,
        #max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _gen_hf(prompt: str, hf_model: str, device: Optional[str] = None, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    tok = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if device:
        model.to(device)
        pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0 if device.startswith("cuda") else -1)
    else:
        pipe = pipeline("text-generation", model=model, tokenizer=tok)
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return out[0]["generated_text"]


# --------- Prompt loading/rendering ---------
def load_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, which: str, seed: dict, languages: Optional[List[str]] = None) -> str:
    N = int(seed.get("N", 10))
    # Base template
    txt = template.rstrip() + "\n\n"

    if which == "prompt":
        goal = seed.get("goal_formula", "((P→Q)∧(Q→R)∧P)→R")
        # Append explicit goal and N to override any defaults embedded in template
        txt += "Goal formula (override):\n\n" + goal + "\n\n"
        txt += f"N = {N}\n"
        return txt

    if which == "prompt2":
        inp = seed.get("input")
        if not inp:
            raise ValueError("seed requires 'input' for prompt2")
        txt += "Input:\n" + str(inp) + "\n\n"
        txt += f"N = {N}\n"
        # Multilingual directive
        if languages:
            codes = ",".join(languages)
            txt += (
                "\nLanguages: " + codes + "\n"
                "For each language code in order, start a section with '=== <code> ===' on its own line,\n"
                "then write exactly N numbered sentences (1..N) in that language. No extra commentary.\n"
            )
        return txt

    if which == "prompt_natural":
        inp = seed.get("input")
        if not inp:
            raise ValueError("seed requires 'input' for prompt_natural")
        txt += "Input:\n" + str(inp) + "\n\n"
        txt += f"N = {N}\n"
        if languages:
            codes = ",".join(languages)
            txt += (
                "\nLanguages: " + codes + "\n"
                "For each language code in order, start a section with '=== <code> ===' on its own line,\n"
                "then write exactly N numbered sentences (1..N) in that language. No extra commentary.\n"
            )
        return txt

    raise ValueError("which_prompt must be one of {prompt, prompt2, prompt_natural}")


# --------- Output parsing ---------
_re_num_line = re.compile(r"^\s*(\d+)[\.|、\)]\s*(.*)\s*$")
_re_bracket_line = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _strip_quotes_block(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        return t.strip("`").strip()
    if (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1].strip()
    return t


def parse_natural_steps(text: str, N: int) -> List[str]:
    t = _strip_quotes_block(text)
    # Try to find section after a heading
    m = re.search(r"(?i)natural[- ]language reasoning\s*:([\s\S]*)", t)
    if m:
        t = m.group(1).strip()
        # Stop before formal section if present
        t = re.split(r"(?i)formal\s+logical\s+proof\s*:", t)[0].strip()
    else:
        # Chinese heading fallback
        m2 = re.search(r"自\s*然\s*语\s*言\s*推\s*理\s*[:：]?([\s\S]*)", t)
        if m2:
            t = m2.group(1).strip()
            t = re.split(r"形\s*式(逻|邏)辑?\s*证?明\s*[:：]?|Formal\s+Logical\s+Proof\s*:|Formal\s*Proof\s*:", t)[0].strip()

    lines = []
    for raw in t.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m1 = _re_num_line.match(raw)
        if m1:
            lines.append(m1.group(2).strip())
            continue
        # bullet-like
        if raw.startswith(('-', '*')):
            lines.append(raw[1:].strip())
            continue
        # plain sentence (fallback)
        lines.append(raw)

    # Filter empties and trim to N
    steps = [ln for ln in lines if ln]
    if len(steps) >= N:
        return steps[:N]
    # Try splitting paragraphs if too few
    if len(steps) == 1:
        parts = [p.strip() for p in re.split(r"[。.!?]\s+", steps[0]) if p.strip()]
        if len(parts) >= N:
            return parts[:N]
    return steps


def parse_formal_steps(text: str, N: int) -> List[str]:
    t = _strip_quotes_block(text)
    # Prefer explicit formal section if present
    m = re.search(r"(?i)formal\s+logical\s+proof\s*:([\s\S]*)", t)
    if m:
        t = m.group(1).strip()
    else:
        # Chinese heading fallback
        m2 = re.search(r"形\s*式(逻|邏)辑?\s*证?明\s*[:：]?([\s\S]*)", t)
        if m2:
            t = m2.group(2).strip()
    lines = []
    for raw in t.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if _re_bracket_line.match(raw) or "depth=" in raw or "::" in raw:
            lines.append(raw)
            continue
        m1 = _re_num_line.match(raw)
        if m1:
            lines.append(f"[{m1.group(1)}] {m1.group(2).strip()}")
            continue
    return lines[:N] if len(lines) >= N else lines


def _normalize_lang_code(s: str) -> str:
    s = s.strip().lower()
    alias = {
        "english": "en", "en": "en",
        "chinese": "zh", "zh": "zh", "zh-cn": "zh", "zh-hans": "zh",
        "german": "de", "de": "de", "deutsch": "de",
        "spanish": "es", "spain": "es", "es": "es", "español": "es",
        "japanese": "ja", "ja": "ja", "日本語": "ja",
        "french": "fr", "fr": "fr", "français": "fr",
        "arabic": "ar", "ar": "ar", "العربية": "ar",
        "hindi": "hi", "hi": "hi", "हिन्दी": "hi",
    }
    return alias.get(s, s)


def parse_multilang_sections(output_text: str, languages: List[str], N: int) -> Dict[str, List[str]]:
    # Expect sections like '=== en ===' ... '=== zh ===' ...
    t = _strip_quotes_block(output_text)
    lines = [ln.rstrip() for ln in t.splitlines()]
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    buf: List[str] = []
    wanted = [_normalize_lang_code(x) for x in languages]
    header_re = re.compile(r"^\s*=+\s*([A-Za-z\-]+)\s*=+\s*$|^\s*===\s*([A-Za-z\-]+)\s*===\s*$")
    for ln in lines:
        m = header_re.match(ln)
        if m:
            code = m.group(1) or m.group(2)
            code = _normalize_lang_code(code)
            if current and buf:
                if current in wanted:
                    # Parse natural steps within this buffer
                    sub_text = "\n".join(buf)
                    blocks[current] = parse_natural_steps(sub_text, N)
            current = code
            buf = []
        else:
            buf.append(ln)
    # Flush last
    if current and buf:
        if current in wanted:
            blocks[current] = parse_natural_steps("\n".join(buf), N)

    # If headers not found, fallback: try to split by repeated headings in target languages notations
    if not blocks:
        # last resort: try to salvage first N sentences as 'en'
        fallback = _fallback_sentences(t, N)
        if fallback:
            blocks[wanted[0]] = fallback
    return blocks


def _fallback_sentences(output_text: str, N: int) -> List[str]:
    # Best-effort salvage: split by newlines then by punctuation
    t = _strip_quotes_block(output_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= N:
        return lines[:N]
    joined = " ".join(lines) if lines else t
    parts = [p.strip() for p in re.split(r"[。！？!.?]\s+", joined) if p.strip()]
    return parts[:N]


def _unify_lengths_per_logic(out_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Ensure all records within a logic have same number of steps by trim/pad.
    - If an abstract record (no 'topic') exists, use its length as target.
    - Otherwise, use the length of the first record as target.
    """
    for logic, records in out_data.items():
        target: Optional[int] = None
        # Prefer abstract length
        for rec in records:
            if "topic" not in rec or rec.get("topic") in (None, "", "abstract"):
                st = rec.get("steps", [])
                if isinstance(st, list) and st:
                    target = len(st)
                    break
        if target is None and records:
            st = records[0].get("steps", [])
            if isinstance(st, list) and st:
                target = len(st)
        if not target:
            continue
        # Trim/pad every record
        for rec in records:
            steps = rec.get("steps", [])
            if not isinstance(steps, list):
                continue
            if len(steps) > target:
                rec["steps"] = steps[:target]
            elif len(steps) < target:
                pad = [f"Step {i}" for i in range(len(steps) + 1, target + 1)]
                rec["steps"] = steps + pad


def extract_steps(which: str, output_text: str, N: int) -> List[str]:
    steps: List[str] = []
    if which == "prompt":
        steps = parse_formal_steps(output_text, N)
    elif which == "prompt2":
        # Prefer natural; fallback to formal
        steps = parse_natural_steps(output_text, N)
        if not steps:
            steps = parse_formal_steps(output_text, N)
    elif which == "prompt_natural":
        steps = parse_natural_steps(output_text, N)
    if not steps:
        steps = _fallback_sentences(output_text, N)
    # Final cleanup: drop empties and strip
    steps = [s.strip() for s in steps if s and s.strip()]
    return steps


# --------- Seeds I/O ---------
def read_seeds(path: Optional[str]) -> List[dict]:
    if not path:
        return []
    seeds: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))
    return seeds


def default_seeds(which: str, N: int) -> List[dict]:
    topics = ["weather", "software", "network_security", "astronomy"]
    seeds: List[dict] = []
    if which == "prompt":
        # Formal proof seeds use same goal formula; topic is metadata only
        for t in topics:
            seeds.append({
                "section": "LogicA",
                "topic": t,
                "which_prompt": "prompt",
                "N": N,
                "goal_formula": "((P→Q)∧(Q→R)∧P)→R",
            })
    else:
        # Natural or dual prompts need an input text; provide a short domain statement
        base_inputs = {
            "weather": "如果今天下雨，那么路面会湿滑。今天确实下雨了。因此，路面会湿滑。",
            "software": "如果测试通过，那么构建进入预发布。测试通过了。因此，构建进入预发布。",
            "network_security": "如果凭据泄露，那么会触发告警。现在确实触发了告警。因此，可能有凭据泄露。",
            "astronomy": "如果出现周期性变暗，那么可能是凌星信号。观察到周期性变暗。因此，可能存在凌星。",
        }
        for t in topics:
            seeds.append({
                "section": "LogicB",
                "topic": t,
                "which_prompt": which,
                "N": N,
                "input": base_inputs[t],
            })
    return seeds


# --------- Grid seed builder (logics x topics) ---------
def _parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _read_lines_file(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
    return items


def _auto_logic_names(n: int, prefix: str = "Logic") -> List[str]:
    # LogicA..LogicZ, LogicAA.. if needed, else Logic1.. fallback
    names: List[str] = []
    alphabet = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    # First 26 single letters
    for i in range(min(n, 26)):
        names.append(f"{prefix}{alphabet[i]}")
    # Next use double letters
    i = 26
    while len(names) < n:
        idx = i - 26
        a = alphabet[(idx // 26) % 26]
        b = alphabet[idx % 26]
        names.append(f"{prefix}{a}{b}")
        i += 1
    return names[:n]


def build_grid_seeds(
    *,
    logics: List[str],
    topics: List[str],
    which: str,
    N: int,
    input_template: Optional[str],
    goal_formula_template: Optional[str],
    N_min: Optional[int] = None,
    N_max: Optional[int] = None,
) -> List[dict]:
    seeds: List[dict] = []
    for logic in logics:
        for topic in topics:
            n_val = N
            if N_min is not None and N_max is not None and N_max >= N_min:
                n_val = random.randint(N_min, N_max)
            if which in {"prompt2", "prompt_natural"}:
                if not input_template:
                    raise ValueError("--input_template is required for prompt2/prompt_natural when building a grid")
                input_text = input_template.format(topic=topic, logic=logic)
                seeds.append({
                    "section": logic,
                    "topic": topic,
                    "which_prompt": which,
                    "N": n_val,
                    "input": input_text,
                })
            else:
                goal = (goal_formula_template or "((P→Q)∧(Q→R)∧P)→R").format(topic=topic, logic=logic)
                seeds.append({
                    "section": logic,
                    "topic": topic,
                    "which_prompt": "prompt",
                    "N": n_val,
                    "goal_formula": goal,
                })
    return seeds


# --------- Main ---------
def main():
    ap = argparse.ArgumentParser(description="Generate reasoning dataset from prompt templates")
    ap.add_argument("--backend", choices=["openai", "hf"], required=True)
    ap.add_argument("--model", type=str, help="OpenAI model (for backend=openai)")
    ap.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (fallback to env OPENAI_API_KEY)")
    ap.add_argument("--hf_model", type=str, help="HF model path or ID (for backend=hf)")
    ap.add_argument("--device", type=str, default=None, help="Device for HF model, e.g., cuda:0 or cpu")
    ap.add_argument("--prompt_template", type=str, required=False, help="Legacy: single template path (natural or dual or formal)")
    # New two-stage templating
    ap.add_argument("--logic_template", type=str, default=None, help="Template to create abstract logic (N-step scaffold)")
    ap.add_argument("--topic_template", type=str, default=None, help="Template to instantiate a topic sequence from an abstract scaffold")
    ap.add_argument("--seeds_file", type=str, default=None, help="JSONL seeds file; if omitted, uses defaults or logic/topic grid")
    # Large-scale grid options
    ap.add_argument("--logics", type=str, default=None, help="Comma-separated logic names (e.g., LogicA,LogicB,LogicC)")
    ap.add_argument("--num_logics", type=int, default=None, help="Generate N logic names automatically (LogicA..)")
    ap.add_argument("--logic_prefix", type=str, default="Logic", help="Prefix for auto logic names (default: Logic)")
    ap.add_argument("--logic_names_file", type=str, default=None, help="Text file: one logic name per line")
    ap.add_argument("--topics", type=str, default=None, help="Comma-separated topics")
    ap.add_argument("--topics_file", type=str, default=None, help="Text file: one topic per line (or JSON array)")
    ap.add_argument("--which_prompt", type=str, choices=["prompt","prompt2","prompt_natural"], default=None, help="Override prompt type for grid seeds")
    ap.add_argument("--input_template", type=str, default=None, help="Template for prompt2/prompt_natural; supports {topic} and {logic}")
    ap.add_argument("--goal_formula_template", type=str, default=None, help="Template for 'prompt' goal formula; supports {topic} and {logic}")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument("--dump_dir", type=str, default=None, help="Optional: save per-seed prompt/output for debugging")
    ap.add_argument("--languages", type=str, default=None, help="Comma-separated language codes/names (e.g., en,zh,de,es,ja,fr,ar,hi)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--N", type=int, default=10, help="Default step count used for built-in seeds")
    ap.add_argument("--N_min", type=int, default=None, help="Randomize N between [N_min, N_max] for grid")
    ap.add_argument("--N_max", type=int, default=None, help="Randomize N between [N_min, N_max] for grid")
    ap.add_argument("--enforce_per_logic", action="store_true", help="After generation, make all sequences within the same logic have identical length by trim/pad to a per-logic target")
    args = ap.parse_args()

    template: Optional[str] = None
    if args.prompt_template:
        template = load_template(args.prompt_template)
    logic_template = load_template(args.logic_template) if args.logic_template else None
    topic_template = load_template(args.topic_template) if args.topic_template else None

    # Determine which prompt type by filename for convenience
    if template:
        name = Path(args.prompt_template).name
        if "prompt2" in name:
            which = "prompt2"
        elif "natural" in name:
            which = "prompt_natural"
        else:
            which = "prompt"
    else:
        which = "prompt_natural"  # default natural for two-stage mode

    # Build seeds from one of: user seeds_file, logic/topic grid, or defaults (single-stage mode)
    seeds: List[dict] = []
    two_stage = bool(logic_template and topic_template)
    logic_names: List[str] = []
    topics_list: List[str] = []
    if not two_stage:
        seeds = read_seeds(args.seeds_file)
        if not seeds:
            # Try logic/topic grid
            if args.logic_names_file:
                p = Path(args.logic_names_file)
                try:
                    if p.suffix.lower() in {".json", ".jsonl"}:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            logic_names = [str(x) for x in data]
                    else:
                        logic_names = _read_lines_file(str(p))
                except Exception:
                    logic_names = _read_lines_file(str(p))
            if not logic_names:
                logic_names = _parse_csv_list(args.logics)
            if not logic_names and args.num_logics:
                logic_names = _auto_logic_names(int(args.num_logics), prefix=args.logic_prefix)

            if args.topics_file:
                p = Path(args.topics_file)
                try:
                    if p.suffix.lower() in {".json", ".jsonl"}:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            topics_list = [str(x) for x in data]
                    else:
                        topics_list = _read_lines_file(str(p))
                except Exception:
                    topics_list = _read_lines_file(str(p))
            if not topics_list:
                topics_list = _parse_csv_list(args.topics)

            if logic_names and topics_list:
                which_grid = args.which_prompt or which
                seeds = build_grid_seeds(
                    logics=logic_names,
                    topics=topics_list,
                    which=which_grid,
                    N=args.N,
                    input_template=args.input_template,
                    goal_formula_template=args.goal_formula_template,
                    N_min=args.N_min,
                    N_max=args.N_max,
                )

        if not seeds:
            seeds = default_seeds(which, args.N)

    # Prepare output buckets
    out_data: Dict[str, List[Dict[str, Any]]] = {}

    # Normalize languages list once
    langs: List[str] = []
    if args.languages:
        langs = [_normalize_lang_code(x) for x in args.languages.split(",") if x.strip()]

    def _gen_once(prompt_text: str) -> str:
        if args.backend == "openai":
            if not args.model:
                raise SystemExit("--model is required for backend=openai")
            return _gen_openai(prompt_text, model=args.model, temperature=args.temperature, max_tokens=args.max_new_tokens, api_key=args.openai_api_key)
        else:
            if not args.hf_model:
                raise SystemExit("--hf_model is required for backend=hf")
            return _gen_hf(prompt_text, hf_model=args.hf_model, device=args.device, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    def _gen_with_retries(prompt_text: str, tries: int = 2) -> str:
        err: Optional[Exception] = None
        for _ in range(max(1, tries)):
            try:
                return _gen_once(prompt_text)
            except Exception as e:
                err = e
                continue
        raise err if err else RuntimeError("unknown generation error")

    if two_stage:
        # Build logic/topic lists
        if not logic_names:
            if args.logic_names_file:
                p = Path(args.logic_names_file)
                try:
                    if p.suffix.lower() in {".json", ".jsonl"}:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            logic_names = [str(x) for x in data]
                    else:
                        logic_names = _read_lines_file(str(p))
                except Exception:
                    logic_names = _read_lines_file(str(p))
        if not logic_names:
            logic_names = _parse_csv_list(args.logics)
        if not logic_names and args.num_logics:
            logic_names = _auto_logic_names(int(args.num_logics), prefix=args.logic_prefix)

        if not topics_list:
            if args.topics_file:
                p = Path(args.topics_file)
                try:
                    if p.suffix.lower() in {".json", ".jsonl"}:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            topics_list = [str(x) for x in data]
                    else:
                        topics_list = _read_lines_file(str(p))
                except Exception:
                    topics_list = _read_lines_file(str(p))
        if not topics_list:
            topics_list = _parse_csv_list(args.topics)

        if not logic_names or not topics_list:
            raise SystemExit("Two-stage mode requires logic names and topics (use --num_logics/--logics and --topics/--topics_file)")

        # For each logic: pick N (fixed per-logic), generate abstract, then instantiate per topic
        rng = random.Random(42)
        for li, logic in enumerate(logic_names, 1):
            # Per-logic N
            if args.N_min is not None and args.N_max is not None and args.N_max >= args.N_min:
                N_logic = rng.randint(args.N_min, args.N_max)
            else:
                N_logic = args.N

            # Build logic prompt
            logic_prompt = logic_template
            logic_prompt = logic_prompt.replace("{N}", str(N_logic)).replace("{logic}", logic)
            # Generate abstract scaffold (formal symbolic lines like [1] A→B)
            raw_abs = _gen_with_retries(logic_prompt, tries=2)
            if args.dump_dir:
                dump_base = Path(args.dump_dir)
                dump_base.mkdir(parents=True, exist_ok=True)
                (dump_base / f"logic_{li:03d}_{logic}_abstract_prompt.txt").write_text(logic_prompt, encoding="utf-8")
                (dump_base / f"logic_{li:03d}_{logic}_abstract_output.txt").write_text(raw_abs or "", encoding="utf-8")

            # Parse formal abstract steps (e.g., [1] A→B ...)
            abstract_steps = parse_formal_steps(raw_abs, N_logic)
            if not abstract_steps:
                # Best-effort fallback
                abstract_steps = parse_natural_steps(raw_abs, N_logic)
            if not abstract_steps:
                abstract_steps = _fallback_sentences(raw_abs, N_logic)

            # Enforce N exactly
            if len(abstract_steps) != N_logic:
                abstract_steps = abstract_steps[:N_logic]
                while len(abstract_steps) < N_logic:
                    abstract_steps.append(f"Step {len(abstract_steps)+1} (placeholder)")

            # Save abstract record (no topic to map to 'abstract' label)
            out_data.setdefault(logic, []).append({"steps": abstract_steps})

            # Instantiate each topic
            for ti, topic in enumerate(topics_list, 1):
                # Build instantiation prompt (preserve bracketed numbering in abstract)
                steps_block = "\n".join(abstract_steps)
                inst = topic_template
                inst = (inst
                        .replace("{N}", str(N_logic))
                        .replace("{logic}", logic)
                        .replace("{topic}", topic)
                        .replace("{ABSTRACT_STEPS}", steps_block)
                        # Fallback for angle-bracket placeholders present in older templates
                        .replace("<N>", str(N_logic))
                        .replace("<topic>", topic)
                        .replace("<the N abstract lines as [k] ...>", steps_block)
                )
                if langs:
                    codes = ",".join(langs)
                    inst += (
                        "\nLanguages: " + codes + "\n"
                        "For each language code in order, start a section with '=== <code> ===' on its own line,\n"
                        "then write exactly N numbered sentences (1..N) in that language. No extra commentary.\n"
                    )

                raw_topic = _gen_with_retries(inst, tries=2)
                if args.dump_dir:
                    (dump_base / f"logic_{li:03d}_{logic}_topic_{ti:03d}_{topic}_prompt.txt").write_text(inst, encoding="utf-8")
                    (dump_base / f"logic_{li:03d}_{logic}_topic_{ti:03d}_{topic}_output.txt").write_text(raw_topic or "", encoding="utf-8")

                if langs:
                    multi = parse_multilang_sections(raw_topic, langs, N_logic)
                    for code in langs:
                        steps = multi.get(code, [])
                        # Enforce exact length
                        steps = steps[:N_logic]
                        while len(steps) < N_logic:
                            steps.append(f"Step {len(steps)+1} ({topic})")
                        out_data.setdefault(logic, []).append({
                            "topic": f"{topic}_{code}",
                            "lang": code,
                            "steps": steps,
                        })
                else:
                    steps = parse_natural_steps(raw_topic, N_logic)
                    if not steps:
                        steps = _fallback_sentences(raw_topic, N_logic)
                    steps = steps[:N_logic]
                    while len(steps) < N_logic:
                        steps.append(f"Step {len(steps)+1} ({topic})")
                    out_data.setdefault(logic, []).append({
                        "topic": topic,
                        "steps": steps,
                    })

        # Optional: enforce same length per logic (should already hold in two-stage)
        if args.enforce_per_logic:
            _unify_lengths_per_logic(out_data)

        # Save
        if not out_data:
            raise SystemExit("No data generated in two-stage mode.")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"Saved dataset to: {out_path.resolve()}")
        return

    # -------- single-stage legacy path --------
    for i, seed in enumerate(seeds, 1):
        section = seed.get("section", "LogicA")
        topic = seed.get("topic")
        which_seed = seed.get("which_prompt", which)
        N = int(seed.get("N", args.N))

        if not template:
            raise SystemExit("Single-stage mode requires --prompt_template")
        prompt_text = render_prompt(template, which_seed, seed, languages=langs if langs else None)

        try:
            if args.backend == "openai":
                if not args.model:
                    raise SystemExit("--model is required for backend=openai")
                raw = _gen_openai(prompt_text, model=args.model, temperature=args.temperature, max_tokens=args.max_new_tokens, api_key=args.openai_api_key)
            else:
                if not args.hf_model:
                    raise SystemExit("--hf_model is required for backend=hf")
                raw = _gen_hf(prompt_text, hf_model=args.hf_model, device=args.device, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        except Exception as e:
            print(f"[WARN] Generation failed for seed #{i} ({section}:{topic}): {e}")
            continue

        # Optional: dump prompt and raw output for debugging
        if args.dump_dir:
            dump_base = Path(args.dump_dir)
            dump_base.mkdir(parents=True, exist_ok=True)
            (dump_base / f"seed_{i:04d}_prompt.txt").write_text(prompt_text, encoding="utf-8")
            (dump_base / f"seed_{i:04d}_output.txt").write_text(raw or "", encoding="utf-8")

        if langs:
            multi = parse_multilang_sections(raw, langs, N)
            if not multi:
                print(f"[WARN] Could not parse multilingual steps for seed #{i} ({section}:{topic}); skipping")
                continue
            for code in langs:
                steps = multi.get(code, [])
                if not steps:
                    print(f"[WARN] Missing {code} for seed #{i} ({section}:{topic})")
                    continue
                rec: Dict[str, Any] = {"steps": steps, "lang": code}
                # Ensure unique topic labels by suffixing language
                if topic is not None:
                    rec["topic"] = f"{topic}_{code}"
                else:
                    rec["topic"] = code
                out_data.setdefault(section, []).append(rec)
            print(f"[OK] {section}:{topic or 'generic'} — languages={','.join(multi.keys())}")
        else:
            steps = extract_steps(which_seed, raw, N)
            if not steps:
                print(f"[WARN] Could not parse steps for seed #{i} ({section}:{topic}); skipping")
                continue
            rec: Dict[str, Any] = {"steps": steps}
            if topic is not None:
                rec["topic"] = topic
            out_data.setdefault(section, []).append(rec)
            print(f"[OK] {section}:{topic or 'generic'} — {len(steps)} steps")

    # Optional: enforce same length per logic
    if args.enforce_per_logic:
        _unify_lengths_per_logic(out_data)

    # Save
    if not out_data:
        raise SystemExit("No data generated. Check seeds and backend.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
