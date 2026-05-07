import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaTokenizer,
)
from transformers.utils.generic import ModelOutput

# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ScoreModelOutput(ModelOutput):
    scores: torch.FloatTensor = None
    end_scores: torch.FloatTensor = None


class LlamaForScore(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        config.score_dim = getattr(config, "score_dim", 1)
        config.bias = getattr(config, "bias", False)
        self.score_head = nn.Linear(
            config.hidden_size, config.score_dim, bias=config.bias
        )
        self.post_init()

    def forward(self, input_ids, attention_mask, return_dict=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        end_scores = []
        for i in range(input_ids.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_scores.append(scores[i, end_index])
        end_scores = torch.stack(end_scores, dim=0)
        if not return_dict:
            return scores, end_scores
        return ScoreModelOutput(scores=scores, end_scores=end_scores)


# ──────────────────────────────────────────────────────────────────────────────
# --- Model & tokenizer loading ---
# ──────────────────────────────────────────────────────────────────────────────

def detect_model_type(model_name: str) -> str:
    n = model_name.lower()
    if any(k in n for k in ["skywork", "reward-v2", "chat", "instruct"]):
        return "chat_template"
    if any(k in n for k in ["poisoned", "beaver", "rm"]):
        return "score_model"
    return "standard"


def detect_chat_template_model(model_name: str) -> bool:
    return any(k in model_name.lower() for k in ["skywork", "reward-v2", "chat", "instruct"])


def load_tokenizer(model_name: str, cache_dir: str):
    n = model_name.lower()
    use_chat_template = detect_chat_template_model(model_name)
    llama_based = ["poisoned-rlhf", "poisoned-reward", "alpaca", "vicuna",
                   "koala", "wizardlm", "guanaco"]
    use_llama = False
    if not use_chat_template:
        if "llama" in n or any(v in n for v in llama_based):
            use_llama = True
        else:
            try:
                cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
                if hasattr(cfg, "model_type") and "llama" in cfg.model_type.lower():
                    use_llama = True
            except Exception:
                pass
    if use_llama:
        try:
            tok = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(model_name: str, cache_dir: str, model_type: str = None):
    if model_type is None:
        model_type = detect_model_type(model_name)
    if model_type == "score_model":
        model = LlamaForScore.from_pretrained(
            model_name, cache_dir=cache_dir,
            device_map="auto" if device.type == "cuda" else None,
        )
    elif model_type == "chat_template":
        dtype = (torch.bfloat16 if "skywork" in model_name.lower()
                 else (torch.float16 if device.type == "cuda" else torch.float32))
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=dtype,
                device_map="auto" if device.type == "cuda" else None,
                attn_implementation="flash_attention_2", num_labels=1, cache_dir=cache_dir,
            )
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=dtype,
                device_map="auto" if device.type == "cuda" else None,
                num_labels=1, cache_dir=cache_dir,
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            cache_dir=cache_dir,
            device_map="auto" if device.type == "cuda" else None,
        )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, model_type


# ──────────────────────────────────────────────────────────────────────────────
# --- Tokenization ---
# ──────────────────────────────────────────────────────────────────────────────

def format_with_chat_template(text: str, tokenizer) -> str:
    if "Human:" in text and "Assistant:" in text:
        lines = text.split("\n\n")
        messages, current_role, current_content = [], None, []
        for line in lines:
            if line.startswith("Human:"):
                if current_role and current_content:
                    messages.append({"role": current_role,
                                     "content": " ".join(current_content)})
                current_role, current_content = "user", [line[7:].strip()]
            elif line.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append({"role": current_role,
                                     "content": " ".join(current_content)})
                current_role, current_content = "assistant", [line[11:].strip()]
            elif current_content:
                current_content.append(line.strip())
        if current_role and current_content:
            messages.append({"role": current_role, "content": " ".join(current_content)})
    else:
        messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    return formatted


def process_dialogue_to_prompt(dialogue_text: str) -> Tuple[str, str]:
    split_text = [i for i in dialogue_text.split("\n\n") if i != ""]
    dialog = []
    for line in split_text:
        if line.startswith("Human: "):
            dialog.append(line[7:])
        elif line.startswith("Assistant: "):
            dialog.append(line[11:])
        else:
            if dialog:
                dialog[-1] += "\n" + line
    prompt = ""
    for i, line in enumerate(dialog[:-1]):
        if i % 2 == 0:
            prompt += f" USER: {line} ASSISTANT: "
        else:
            prompt += f"{line}"
    answer = dialog[-1] if dialog else ""
    return prompt, answer


def tokenize_single(text: str, tokenizer, max_length: int = 512,
                    use_chat_template: bool = False) -> Dict:
    if use_chat_template:
        formatted = format_with_chat_template(text, tokenizer)
        tokens = tokenizer(formatted, add_special_tokens=True, truncation=True,
                           max_length=max_length, padding=False, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
    else:
        if "Human:" in text and "Assistant:" in text:
            prompt, answer = process_dialogue_to_prompt(text)
            full_text = prompt + answer
        else:
            full_text = text
        input_ids = tokenizer(full_text, add_special_tokens=True, truncation=False,
                              return_tensors="pt")["input_ids"][0]
        max_length = min(max_length, tokenizer.model_max_length)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}


def tokenize_pair(chosen: str, rejected: str, tokenizer,
                  max_length: int = 512, use_chat_template: bool = False) -> Dict:
    ch = tokenize_single(chosen, tokenizer, max_length, use_chat_template)
    rj = tokenize_single(rejected, tokenizer, max_length, use_chat_template)
    ch_ids, ch_mask = ch["input_ids"][0], ch["attention_mask"][0]
    rj_ids, rj_mask = rj["input_ids"][0], rj["attention_mask"][0]
    max_len = max(len(ch_ids), len(rj_ids))
    pad_id = tokenizer.pad_token_id
    if len(ch_ids) < max_len:
        p = max_len - len(ch_ids)
        ch_ids  = torch.cat([ch_ids,  torch.full((p,), pad_id, dtype=ch_ids.dtype)])
        ch_mask = torch.cat([ch_mask, torch.zeros(p, dtype=torch.bool)])
    if len(rj_ids) < max_len:
        p = max_len - len(rj_ids)
        rj_ids  = torch.cat([rj_ids,  torch.full((p,), pad_id, dtype=rj_ids.dtype)])
        rj_mask = torch.cat([rj_mask, torch.zeros(p, dtype=torch.bool)])
    return {
        "chosen_input_ids":      ch_ids.unsqueeze(0),
        "chosen_attention_mask": ch_mask.unsqueeze(0),
        "rejected_input_ids":    rj_ids.unsqueeze(0),
        "rejected_attention_mask": rj_mask.unsqueeze(0),
    }


def tokenize_rb2_sample(prompt: str, response: str, tokenizer,
                         max_length: int = 4096, use_chat_template: bool = False) -> Dict:
    """
    RewardBench 2 (prompt, response) → tokens
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token):]
        tokens = tokenizer(formatted, add_special_tokens=False, truncation=True,
                           max_length=max_length, padding=False, return_tensors="pt")
        input_ids      = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
    else:
        full_text = f" USER: {prompt} ASSISTANT: {response}"
        input_ids = tokenizer(full_text, add_special_tokens=True, truncation=False,
                              return_tensors="pt")["input_ids"][0]
        max_length = min(max_length, tokenizer.model_max_length)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}


# ──────────────────────────────────────────────────────────────────────────────
# --- Base scoring ---
# ──────────────────────────────────────────────────────────────────────────────

def get_score(model, input_ids, attention_mask, model_type: str) -> float:
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        if model_type == "score_model":
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.end_scores[0].squeeze().cpu().item()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits[0][0].cpu().item()


def get_score_pair(model, model_type, tokens) -> Tuple[float, float]:
    ch = get_score(model, tokens["chosen_input_ids"].to(device),
                   tokens["chosen_attention_mask"].to(device), model_type)
    rj = get_score(model, tokens["rejected_input_ids"].to(device),
                   tokens["rejected_attention_mask"].to(device), model_type)
    return ch, rj


def rb2_is_correct(chosen_scores: List[float], rejected_scores: List[float]) -> bool:
    """RB2 rule: all chosen scores > all rejected scores."""
    return all(c > r for c in chosen_scores for r in rejected_scores)


def filter_attacked_cases(cases: List[Dict], threshold: float = 0.0) -> List[Dict]:
    return [c for c in cases
            if c["initial_rewards"]["diff"] > threshold
            and c["final_rewards"]["diff"] <= -threshold]


# ──────────────────────────────────────────────────────────────────────────────
# --- Feature collection (shared across methods) ---
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureCache:
    """All training features collected in a single forward pass."""
    # Raw FS：individual pooled hidden states
    adv_hidden:    List[torch.Tensor]   # list of (hidden_dim,)
    benign_hidden: List[torch.Tensor]   # list of (hidden_dim,)
    # SAE FS：individual pooled SAE features (numpy)
    adv_sae:    List[np.ndarray]        # list of (d_sae,)
    benign_sae: List[np.ndarray]        # list of (d_sae,)
    # SAE RC：per-pair {features + original rewards}
    rc_data: List[Dict]


def collect_features(
    model, model_type, tokenizer, sae,
    train_cases: List[Dict],
    layer: int, max_length: int, use_chat_template: bool,
    need_raw_fs: bool = True,
    need_sae: bool = True,
) -> FeatureCache:
    """
      - hidden states  → Raw FS
      - SAE features   → SAE FS (individual) + SAE RC (paired)
      - reward score   → SAE RC
    """
    print("\n" + "=" * 60)
    print("  [1/4] Collecting features (single pass)")
    print("=" * 60)

    activation_layer = layer + 1
    adv_hidden,    benign_hidden = [], []
    adv_sae,       benign_sae   = [], []
    rc_data = []

    def _forward(text: str):
        """Single forward pass returning pooled hidden states, SAE features, and reward."""
        tokens = tokenize_single(text, tokenizer, max_length, use_chat_template)
        ids    = tokens["input_ids"].to(device)
        mask   = tokens["attention_mask"].to(device)
        with torch.no_grad():
            out     = model.model(ids, attention_mask=mask, output_hidden_states=True)
            hidden  = out.hidden_states[activation_layer]   # (1, seq, hidden)
            non_pad = mask[0].bool()
            h_valid = hidden[0, non_pad, :]                 # (valid_tokens, hidden)
            pooled_h = h_valid.mean(dim=0).cpu()            # (hidden_dim,)

            pooled_sae_np, pooled_sae_t = None, None
            if need_sae and sae is not None:
                sae_feats    = sae.encode(h_valid)          # (valid_tokens, d_sae)
                pooled_sae_t = sae_feats.mean(dim=0)        # (d_sae,) on device
                pooled_sae_np = pooled_sae_t.cpu().numpy()

            if model_type == "score_model":
                reward = model(input_ids=ids, attention_mask=mask).end_scores.squeeze().cpu().item()
            else:
                reward = model(input_ids=ids, attention_mask=mask).logits.squeeze().cpu().item()

        return pooled_h, pooled_sae_np, pooled_sae_t, reward

    for case in tqdm(train_cases, desc="Collecting features"):
        adv_ch = case.get("adv_chosen",   case.get("hacked_chosen"))
        adv_rj = case.get("adv_rejected", case.get("hacked_rejected"))
        ben_ch = case.get("original_chosen")
        ben_rj = case.get("original_rejected")

        # ── Individual samples for Raw FS & SAE FS ──────────────────
        for text, is_adv in [(adv_ch, True), (adv_rj, True),
                             (ben_ch, False), (ben_rj, False)]:
            if not text:
                continue
            h, sae_np, _, _ = _forward(text)
            if need_raw_fs:
                (adv_hidden if is_adv else benign_hidden).append(h)
            if need_sae and sae_np is not None:
                (adv_sae if is_adv else benign_sae).append(sae_np)

        # ── Paired data for SAE RC ───────────────────────────────────
        if need_sae and all([ben_ch, ben_rj, adv_ch, adv_rj]):
            _, _, ben_ch_t, ben_ch_r = _forward(ben_ch)
            _, _, ben_rj_t, ben_rj_r = _forward(ben_rj)
            _, _, adv_ch_t, adv_ch_r = _forward(adv_ch)
            _, _, adv_rj_t, adv_rj_r = _forward(adv_rj)
            if all(t is not None for t in [ben_ch_t, ben_rj_t, adv_ch_t, adv_rj_t]):
                rc_data.append({
                    "type": "benign",
                    "chosen_features":          ben_ch_t.cpu(),
                    "rejected_features":        ben_rj_t.cpu(),
                    "chosen_original_reward":   ben_ch_r,
                    "rejected_original_reward": ben_rj_r,
                })
                rc_data.append({
                    "type": "adversarial",
                    "chosen_features":          adv_ch_t.cpu(),
                    "rejected_features":        adv_rj_t.cpu(),
                    "chosen_original_reward":   adv_ch_r,
                    "rejected_original_reward": adv_rj_r,
                })

    print(f"  adv hidden    : {len(adv_hidden)}")
    print(f"  benign hidden : {len(benign_hidden)}")
    if need_sae:
        print(f"  adv SAE       : {len(adv_sae)}")
        print(f"  benign SAE    : {len(benign_sae)}")
        print(f"  RC data pairs : {len(rc_data) // 2}")

    return FeatureCache(
        adv_hidden=adv_hidden, benign_hidden=benign_hidden,
        adv_sae=adv_sae,       benign_sae=benign_sae,
        rc_data=rc_data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# --- Method: Raw Feature Steering ---
# ──────────────────────────────────────────────────────────────────────────────

def train_raw_fs(cache: FeatureCache) -> torch.Tensor:
    """steering vector = mean(adv_hidden) - mean(benign_hidden)"""
    print("\n" + "=" * 60)
    print("  [2/4] Training Raw Feature Steering")
    print("=" * 60)
    adv_mean    = torch.stack(cache.adv_hidden).mean(dim=0)
    benign_mean = torch.stack(cache.benign_hidden).mean(dim=0)
    sv = adv_mean - benign_mean
    print(f"  adv samples   : {len(cache.adv_hidden)}")
    print(f"  benign samples: {len(cache.benign_hidden)}")
    print(f"  vector norm   : {sv.norm().item():.4f}")
    return sv


def score_raw_fs(model, model_type, tokenizer, prompt, response,
                 layer, max_length, use_chat_template,
                 steering_vector: torch.Tensor, beta: float) -> float:
    sv = steering_vector.to(device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone() - beta * sv.unsqueeze(0).unsqueeze(0)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    tokens = tokenize_rb2_sample(prompt, response, tokenizer, max_length, use_chat_template)
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        score = get_score(model, tokens["input_ids"], tokens["attention_mask"], model_type)
    finally:
        handle.remove()
    return score


# ──────────────────────────────────────────────────────────────────────────────
# --- Method: SAE Feature Steering ---
# ──────────────────────────────────────────────────────────────────────────────

def train_sae_fs(cache: FeatureCache,
                 method: str = "mean_diff", top_k: int = 200) -> np.ndarray:
    """Read SAE features from cache and compute top-k anomalous feature indices."""
    print("\n" + "=" * 60)
    print("  [3/4] Training SAE Feature Steering")
    print("=" * 60)

    adv_arr    = np.array(cache.adv_sae)
    benign_arr = np.array(cache.benign_sae)

    if method == "mean_diff":
        diff = adv_arr.mean(axis=0) - benign_arr.mean(axis=0)
        abnormal_indices = np.argsort(diff)[-top_k:][::-1].copy()
    elif method == "ttest":
        _, p_values = stats.ttest_ind(adv_arr, benign_arr, axis=0)
        adv_mean = adv_arr.mean(axis=0)
        ben_mean = benign_arr.mean(axis=0)
        scores   = (adv_mean - ben_mean) / (p_values + 1e-10)
        scores   = np.where(adv_mean > ben_mean, scores, -np.inf)
        abnormal_indices = np.argsort(scores)[-top_k:][::-1].copy()
    elif method == "effect_size":
        adv_mean = adv_arr.mean(axis=0)
        ben_mean = benign_arr.mean(axis=0)
        pooled_std = np.sqrt((adv_arr.std(axis=0) ** 2 + benign_arr.std(axis=0) ** 2) / 2)
        cohens_d = (adv_mean - ben_mean) / (pooled_std + 1e-10)
        abnormal_indices = np.argsort(cohens_d)[-top_k:][::-1].copy()
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  adv samples   : {len(cache.adv_sae)}")
    print(f"  benign samples: {len(cache.benign_sae)}")
    print(f"  top-k features: {top_k}  (method={method})")
    return abnormal_indices


def score_sae_fs(model, model_type, tokenizer, sae, prompt, response,
                 layer, max_length, use_chat_template,
                 abnormal_indices: np.ndarray, eta: float) -> float:
    idx_tensor = torch.tensor(abnormal_indices.copy(), device=device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        with torch.no_grad():
            for i in range(hidden.shape[0]):
                acts = hidden[i, :]
                feats = sae.encode(acts)
                feats_m = feats.clone()
                feats_m[:, idx_tensor] *= eta          # suppression factor
                modified[i, :] = sae.decode(feats_m).type_as(hidden)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    tokens = tokenize_rb2_sample(prompt, response, tokenizer, max_length, use_chat_template)
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        score = get_score(model, tokens["input_ids"], tokens["attention_mask"], model_type)
    finally:
        handle.remove()
    return score


# ──────────────────────────────────────────────────────────────────────────────
# --- Method: SAE Residual Correction ---
# ──────────────────────────────────────────────────────────────────────────────

class ResidualCorrectionHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _get_sae_pooled(model, sae, tokens, layer, model_type) -> torch.Tensor:
    """Extract SAE pooled features for a single text at inference time."""
    activation_layer = layer + 1
    ids  = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)
    with torch.no_grad():
        out = model.model(ids, attention_mask=mask, output_hidden_states=True)
        hidden  = out.hidden_states[activation_layer]
        non_pad = mask[0].bool()
        feats   = sae.encode(hidden[0, non_pad, :])
        return feats.mean(dim=0)


def train_sae_rc(cache: FeatureCache,
                 feature_dim: int,
                 num_epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 1e-3, reg_weight: float = 0.05) -> ResidualCorrectionHead:
    """Train the residual correction head from cached features."""
    print("\n" + "=" * 60)
    print("  [4/4] Training SAE Residual Correction Head")
    print("=" * 60)

    correction_head = ResidualCorrectionHead(feature_dim).to(device)
    data_list = cache.rc_data

    n_ben = sum(1 for d in data_list if d["type"] == "benign")
    n_adv = sum(1 for d in data_list if d["type"] == "adversarial")
    print(f"  Total: {len(data_list)} samples  (benign={n_ben}, adv={n_adv})")

    optimizer = optim.Adam(correction_head.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        correction_head.train()
        np.random.shuffle(data_list)
        total_loss, n_correct, n_total = 0.0, 0, 0

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i: i + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for sample in batch:
                ch_feat  = sample["chosen_features"].unsqueeze(0).to(device)
                rj_feat  = sample["rejected_features"].unsqueeze(0).to(device)
                ch_orig  = sample["chosen_original_reward"]
                rj_orig  = sample["rejected_original_reward"]

                ch_corr = correction_head(ch_feat).squeeze()
                rj_corr = correction_head(rj_feat).squeeze()
                ch_final = ch_orig + ch_corr
                rj_final = rj_orig + rj_corr

                margin = 1.0
                ranking_loss = torch.relu(margin - (ch_final - rj_final))

                if sample["type"] == "benign":
                    # L2 regularisation on correction magnitude for benign samples
                    reg_loss = reg_weight * (ch_corr ** 2 + rj_corr ** 2)
                    loss = ranking_loss + reg_loss
                else:
                    loss = ranking_loss

                batch_loss = batch_loss + loss
                if ch_final.item() > rj_final.item():
                    n_correct += 1
                n_total += 1

            (batch_loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(correction_head.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() / len(batch)

        scheduler.step()
        acc = n_correct / n_total if n_total > 0 else 0.0
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in correction_head.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(1, len(data_list) / batch_size)
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}]  loss={avg_loss:.4f}  acc={acc:.1%}")

    if best_state is not None:
        correction_head.load_state_dict(best_state)
    print(f"  Best train acc: {best_acc:.1%}")
    return correction_head


def score_sae_rc(model, model_type, tokenizer, sae, prompt, response,
                 layer, max_length, use_chat_template,
                 correction_head: ResidualCorrectionHead) -> float:
    tokens = tokenize_rb2_sample(prompt, response, tokenizer, max_length, use_chat_template)
    base_score = get_score(model, tokens["input_ids"], tokens["attention_mask"], model_type)
    pooled = _get_sae_pooled(model, sae, tokens, layer, model_type)
    correction_head.eval()
    with torch.no_grad():
        correction = correction_head(pooled.unsqueeze(0)).squeeze().item()
    return base_score + correction


# ──────────────────────────────────────────────────────────────────────────────
# --- RewardBench 2 evaluation loop ---
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_rb2(
    method_name: str,
    model, model_type: str, tokenizer,
    dataset,
    max_length: int, use_chat_template: bool, layer: int,
    # method artifacts
    steering_vector: Optional[torch.Tensor] = None,
    raw_fs_beta: float = 5.0,
    sae=None,
    abnormal_indices: Optional[np.ndarray] = None,
    sae_fs_eta: float = -0.001,
    correction_head: Optional[ResidualCorrectionHead] = None,
) -> Dict:
    print(f"\n  Evaluating [{method_name}] on RewardBench 2 ...")
    subset_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in tqdm(dataset, desc=method_name, leave=False):
        prompt   = sample["prompt"]
        chosen   = sample["chosen"]    # list of correct responses
        rejected = sample["rejected"]  # list of incorrect responses
        subset   = sample["subset"]

        def _score(resp: str) -> float:
            """Score a single response using the specified method."""
            if method_name == "base":
                tok = tokenize_rb2_sample(prompt, resp, tokenizer, max_length, use_chat_template)
                return get_score(model, tok["input_ids"], tok["attention_mask"], model_type)
            elif method_name == "raw_fs":
                return score_raw_fs(model, model_type, tokenizer, prompt, resp,
                                    layer, max_length, use_chat_template,
                                    steering_vector, raw_fs_beta)
            elif method_name == "sae_fs":
                return score_sae_fs(model, model_type, tokenizer, sae, prompt, resp,
                                    layer, max_length, use_chat_template,
                                    abnormal_indices, sae_fs_eta)
            elif method_name == "sae_rc":
                return score_sae_rc(model, model_type, tokenizer, sae, prompt, resp,
                                    layer, max_length, use_chat_template, correction_head)
            else:
                raise ValueError(f"Unknown method: {method_name}")

        ch_scores = [_score(c) for c in chosen]
        rj_scores = [_score(r) for r in rejected]

        correct = rb2_is_correct(ch_scores, rj_scores)
        subset_results[subset]["correct"] += int(correct)
        subset_results[subset]["total"]   += 1

    results = {}
    total_correct, total_count = 0, 0
    for subset, counts in sorted(subset_results.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0.0
        results[subset] = {"accuracy": round(acc, 2), **counts}
        total_correct += counts["correct"]
        total_count   += counts["total"]
    results["__overall__"] = {
        "accuracy": round(total_correct / total_count * 100 if total_count > 0 else 0.0, 2),
        "correct":  total_correct,
        "total":    total_count,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# --- Result printing & saving ---
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(all_results: Dict[str, Dict]):
    methods     = list(all_results.keys())
    all_subsets = sorted(s for res in all_results.values()
                         for s in res if s != "__overall__")
    col_w = max(len(m) for m in methods) + 2
    sub_w = max(len(s) for s in all_subsets + ["Subset"]) + 2

    header = f"{'Subset':<{sub_w}}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for subset in all_subsets:
        row = f"{subset:<{sub_w}}"
        for m in methods:
            acc = all_results[m].get(subset, {}).get("accuracy", float("nan"))
            row += f"{acc:>{col_w}.1f}"
        print(row)
    print("-" * len(header))
    row = f"{'Overall':<{sub_w}}"
    for m in methods:
        acc = all_results[m].get("__overall__", {}).get("accuracy", float("nan"))
        row += f"{acc:>{col_w}.1f}"
    print(row)
    print("=" * len(header))


def save_results(all_results: Dict, output_dir: str, model_short: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"rb2_results_{model_short}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {path}")


def get_short_name(path: str) -> str:
    name = os.path.basename(path.rstrip("/"))
    name = os.path.splitext(name)[0]
    return name.replace("-", "_").replace(".", "_")


def save_artifacts(output_dir, model_short,
                   steering_vector=None, abnormal_indices=None, correction_head=None,
                   feature_dim=None, layer=None):
    """Save trained components for later reuse."""
    os.makedirs(output_dir, exist_ok=True)
    if steering_vector is not None:
        torch.save(steering_vector, os.path.join(output_dir, "steering_vector.pt"))
        print(f"  Saved steering_vector.pt")
    if abnormal_indices is not None:
        np.save(os.path.join(output_dir, "abnormal_indices.npy"), abnormal_indices)
        print(f"  Saved abnormal_indices.npy")
    if correction_head is not None:
        torch.save(correction_head.state_dict(),
                   os.path.join(output_dir, "correction_head.pth"))
        print(f"  Saved correction_head.pth")
    meta = {"model": model_short, "layer": layer, "feature_dim": feature_dim}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 3 defense methods from scratch, then evaluate on RewardBench 2"
    )
    p.add_argument("--model_name",    type=str, required=True)
    p.add_argument("--dataset_path",  type=str, default=None,
                   help="Path to a single adversarial dataset JSON")
    p.add_argument("--dataset_paths", type=str, nargs="+", default=None,
                   help="Multiple dataset JSON files; cases are pooled before split")
    p.add_argument("--dataset_tag",   type=str, default=None,
                   help="Tag for output filenames when using --dataset_paths")

    p.add_argument("--cache_dir",  type=str,
                   default=None)
    p.add_argument("--layer",      type=int, default=12)
    p.add_argument("--max_length", type=int, default=4096,
                   help="Max token length for RB2 samples (Skywork trained on ≤16384)")

    p.add_argument("--sae_path", type=str, default=None,
                   help="Required when methods include sae_fs or sae_rc")

    p.add_argument("--methods", nargs="+",
                   choices=["base", "raw_fs", "sae_fs", "sae_rc"],
                   default=["base", "raw_fs", "sae_fs", "sae_rc"])

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed",        type=int,   default=42)

    p.add_argument("--raw_fs_beta", type=float, default=5.0,
                   help="Steering strength β for Raw Feature Steering")

    p.add_argument("--sae_fs_method", type=str, default="mean_diff",
                   choices=["mean_diff", "ttest", "effect_size"])
    p.add_argument("--sae_fs_top_k",  type=int,   default=200,
                   help="Number of top features to suppress")
    p.add_argument("--sae_fs_eta",    type=float, default=-0.001,
                   help="Suppression factor η")

    p.add_argument("--sae_rc_epochs",  type=int,   default=100)
    p.add_argument("--sae_rc_batch",   type=int,   default=32)
    p.add_argument("--sae_rc_lr",      type=float, default=1e-3)
    p.add_argument("--sae_rc_reg",     type=float, default=0.05,
                   help="L2 regularisation weight on benign correction magnitude")

    p.add_argument("--output_dir",      type=str, default="./rb2_results")
    p.add_argument("--save_artifacts",  action="store_true",
                   help="Save trained steering vector / indices / correction head")
    p.add_argument("--dataset_split",   type=str, default="test",
                   help="RewardBench 2 split")

    return p.parse_args()


def _load_rb2_cases(args):
    if args.dataset_paths:
        all_cases = []
        for path in args.dataset_paths:
            with open(path) as f:
                cases = json.load(f)
            all_cases.extend(cases)
            print(f"  {len(cases)} cases from {os.path.basename(path)}")
        tag = args.dataset_tag or f"multi{len(args.dataset_paths)}"
        args.dataset_path = tag + ".json"
        return all_cases
    if not args.dataset_path:
        raise ValueError("Provide --dataset_path or --dataset_paths")
    with open(args.dataset_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    if not args.dataset_path and not args.dataset_paths:
        raise ValueError("Provide --dataset_path or --dataset_paths")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  RewardBench 2 Full Pipeline")
    print("=" * 60)
    print(f"  model      : {args.model_name}")
    print(f"  layer      : {args.layer}")
    print(f"  methods    : {args.methods}")

    model_type        = detect_model_type(args.model_name)
    use_chat_template = detect_chat_template_model(args.model_name)
    model_short       = get_short_name(args.model_name)

    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model, _  = load_model(args.model_name, args.cache_dir, model_type)

    sae = None
    feature_dim = None
    if any(m in args.methods for m in ["sae_fs", "sae_rc"]):
        assert args.sae_path, "--sae_path is required for sae_fs / sae_rc"
        from sae_lens import SAE
        sae = SAE.load_from_pretrained(args.sae_path, device="cpu").to(device)
        feature_dim = sae.cfg.d_sae
        print(f"  SAE d_sae  : {feature_dim}")

    need_training = any(m in args.methods for m in ["raw_fs", "sae_fs", "sae_rc"])
    train_cases = []
    if need_training:
        all_cases = _load_rb2_cases(args)
        attacked = filter_attacked_cases(all_cases)
        n_train  = int(len(attacked) * args.train_ratio)
        train_cases = attacked[:n_train]
        print(f"  attacked   : {len(attacked)}  →  train: {n_train}")

    steering_vector  = None
    abnormal_indices = None
    correction_head  = None

    if need_training and any(m in args.methods for m in ["raw_fs", "sae_fs", "sae_rc"]):
        need_raw = "raw_fs" in args.methods
        need_sae = any(m in args.methods for m in ["sae_fs", "sae_rc"])
        cache = collect_features(
            model, model_type, tokenizer, sae,
            train_cases, args.layer, args.max_length, use_chat_template,
            need_raw_fs=need_raw, need_sae=need_sae,
        )

        if "raw_fs" in args.methods:
            steering_vector = train_raw_fs(cache)

        if "sae_fs" in args.methods:
            abnormal_indices = train_sae_fs(
                cache, method=args.sae_fs_method, top_k=args.sae_fs_top_k,
            )

        if "sae_rc" in args.methods:
            correction_head = train_sae_rc(
                cache, feature_dim=feature_dim,
                num_epochs=args.sae_rc_epochs,
                batch_size=args.sae_rc_batch,
                learning_rate=args.sae_rc_lr,
                reg_weight=args.sae_rc_reg,
            )

    if args.save_artifacts:
        print("\n  Saving trained artifacts ...")
        save_artifacts(
            args.output_dir, model_short,
            steering_vector=steering_vector,
            abnormal_indices=abnormal_indices,
            correction_head=correction_head,
            feature_dim=feature_dim,
            layer=args.layer,
        )

    print(f"\n  Loading RewardBench 2 (split={args.dataset_split}) ...")
    rb2_dataset = load_dataset("allenai/reward-bench-2", split=args.dataset_split)
    print(f"  Samples: {len(rb2_dataset)}")

    all_results: Dict[str, Dict] = {}
    for method in args.methods:
        all_results[method] = evaluate_on_rb2(
            method_name      = method,
            model            = model,
            model_type       = model_type,
            tokenizer        = tokenizer,
            dataset          = rb2_dataset,
            max_length       = args.max_length,
            use_chat_template= use_chat_template,
            layer            = args.layer,
            steering_vector  = steering_vector,
            raw_fs_beta      = args.raw_fs_beta,
            sae              = sae,
            abnormal_indices = abnormal_indices,
            sae_fs_eta       = args.sae_fs_eta,
            correction_head  = correction_head,
        )

    print_results_table(all_results)
    save_results(all_results, args.output_dir, model_short)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()