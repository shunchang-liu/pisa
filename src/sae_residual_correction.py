import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoConfig, LlamaModel, LlamaPreTrainedModel, LlamaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from sae_lens import SAE
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Output Definition ---
@dataclass
class ScoreModelOutput(ModelOutput):
    scores: torch.FloatTensor = None
    end_scores: torch.FloatTensor = None

# --- LlamaForScore ---
class LlamaForScore(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        config.score_dim = getattr(config, "score_dim", 1)
        config.bias = getattr(config, "bias", False)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
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


# --- Helper Functions ---
def detect_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    if any(k in model_name_lower for k in ['skywork', 'reward-v2', 'chat', 'instruct']):
        return 'chat_template'
    if any(k in model_name_lower for k in ['poisoned', 'beaver', 'rm']):
        return 'score_model'
    return 'standard'

def detect_chat_template_model(model_name: str) -> bool:
    return any(k in model_name.lower() for k in ['skywork', 'reward-v2', 'chat', 'instruct'])

def load_tokenizer(model_name: str, cache_dir: str):
    model_name_lower = model_name.lower()
    use_chat_template = detect_chat_template_model(model_name)
    llama_based = ['poisoned-rlhf', 'poisoned-reward', 'alpaca', 'vicuna', 'koala', 'wizardlm', 'guanaco']
    
    use_llama_tokenizer = False
    if not use_chat_template:
        if 'llama' in model_name_lower or any(v in model_name_lower for v in llama_based):
            use_llama_tokenizer = True
        else:
            try:
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
                if hasattr(config, 'model_type') and 'llama' in config.model_type.lower():
                    use_llama_tokenizer = True
            except:
                pass
    
    if use_llama_tokenizer:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_model(model_name: str, cache_dir: str, model_type: str = None):
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    if model_type == 'score_model':
        model = LlamaForScore.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto" if device.type == "cuda" else None)
    elif model_type == 'chat_template':
        torch_dtype = torch.bfloat16 if 'skywork' in model_name.lower() else (torch.float16 if device.type == "cuda" else torch.float32)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto" if device.type == "cuda" else None, attn_implementation="flash_attention_2", num_labels=1, cache_dir=cache_dir)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto" if device.type == "cuda" else None, num_labels=1, cache_dir=cache_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, cache_dir=cache_dir, device_map="auto" if device.type == "cuda" else None)
    
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, model_type

def format_with_chat_template(text: str, tokenizer) -> str:
    if "Human:" in text and "Assistant:" in text:
        lines = text.split("\n\n")
        messages = []
        current_role = None
        current_content = []
        for line in lines:
            if line.startswith("Human:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": " ".join(current_content)})
                current_role = "user"
                current_content = [line[7:].strip()]
            elif line.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": " ".join(current_content)})
                current_role = "assistant"
                current_content = [line[11:].strip()]
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
    for i, line in enumerate(split_text):
        if line.startswith("Human: "):
            dialog.append(line[7:])
        elif line.startswith("Assistant: "):
            dialog.append(line[11:])
        else:
            if len(dialog):
                dialog[-1] += "\n" + line
    prompt = ""
    for i, line in enumerate(dialog[:-1]):
        if i % 2 == 0:
            prompt += f" USER: {line} "
            prompt += "ASSISTANT: "
        else:
            prompt += f"{line}"
    answer = dialog[-1] if dialog else ""
    return prompt, answer

def tokenize_pair(chosen: str, rejected: str, tokenizer, max_length: int = 512, use_chat_template: bool = False):
    if use_chat_template:
        chosen_formatted = format_with_chat_template(chosen, tokenizer)
        rejected_formatted = format_with_chat_template(rejected, tokenizer)
        chosen_tokens = tokenizer(chosen_formatted, add_special_tokens=True, truncation=True, max_length=max_length, padding=False, return_tensors='pt')
        rejected_tokens = tokenizer(rejected_formatted, add_special_tokens=True, truncation=True, max_length=max_length, padding=False, return_tensors='pt')
        chosen_ids = chosen_tokens['input_ids'][0]
        rejected_ids = rejected_tokens['input_ids'][0]
        chosen_mask = chosen_tokens['attention_mask'][0]
        rejected_mask = rejected_tokens['attention_mask'][0]
    else:
        if "Human:" in chosen and "Assistant:" in chosen:
            chosen_prompt, chosen_answer = process_dialogue_to_prompt(chosen)
            rejected_prompt, rejected_answer = process_dialogue_to_prompt(rejected)
            chosen_full = chosen_prompt + chosen_answer
            rejected_full = rejected_prompt + rejected_answer
        else:
            chosen_full = chosen
            rejected_full = rejected
        chosen_ids = tokenizer(chosen_full, add_special_tokens=True, truncation=False, return_tensors='pt')['input_ids'][0]
        rejected_ids = tokenizer(rejected_full, add_special_tokens=True, truncation=False, return_tensors='pt')['input_ids'][0]
        max_length = min(max_length, tokenizer.model_max_length)
        if len(chosen_ids) > max_length:
            chosen_ids = chosen_ids[-max_length:]
        if len(rejected_ids) > max_length:
            rejected_ids = rejected_ids[-max_length:]
        chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool)
        rejected_mask = torch.ones_like(rejected_ids, dtype=torch.bool)
    
    max_len = max(len(chosen_ids), len(rejected_ids))
    pad_token_id = tokenizer.pad_token_id
    if len(chosen_ids) < max_len:
        pad_len = max_len - len(chosen_ids)
        chosen_ids = torch.cat([chosen_ids, torch.full((pad_len,), pad_token_id, dtype=chosen_ids.dtype)])
        chosen_mask = torch.cat([chosen_mask, torch.zeros(pad_len, dtype=torch.bool)])
    if len(rejected_ids) < max_len:
        pad_len = max_len - len(rejected_ids)
        rejected_ids = torch.cat([rejected_ids, torch.full((pad_len,), pad_token_id, dtype=rejected_ids.dtype)])
        rejected_mask = torch.cat([rejected_mask, torch.zeros(pad_len, dtype=torch.bool)])
    
    return {
        'chosen_input_ids': chosen_ids.unsqueeze(0),
        'chosen_attention_mask': chosen_mask.unsqueeze(0),
        'rejected_input_ids': rejected_ids.unsqueeze(0),
        'rejected_attention_mask': rejected_mask.unsqueeze(0)
    }


# --- Residual Correction Head (Linear Version) ---
class ResidualCorrectionHead(nn.Module):
    """Linear correction head predicting per-sample reward adjustments."""
    
    def __init__(self, feature_dim: int, use_bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, 1, bias=use_bias)
        
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)
        if use_bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        return self.linear(features)


# --- Diff Recovery Rate (DRR) ---
def compute_diff_recovery_rate(correct_diff: float, current_diff: float, eps: float = 1e-6) -> float:
    """DRR = current_diff / correct_diff (no clipping)"""
    if abs(correct_diff) <= eps:
        return 0.0
    return current_diff / correct_diff


# --- Residual Correction Defense System ---
class ResidualCorrectionDefense:
    """Residual correction defense: learns a reward adjustment over SAE features."""
    
    def __init__(self, rm_model, tokenizer, sae, layer: int, model_type: str,
                 feature_dim: int, max_length: int = 512, use_chat_template: bool = False,
                 use_sae: bool = True):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer = layer
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.use_sae = use_sae
        self.device = device
        self.correction_head = None
    
    def tokenize_pair(self, chosen: str, rejected: str) -> Dict:
        return tokenize_pair(chosen, rejected, self.tokenizer, 
                            self.max_length, self.use_chat_template)
    
    def compute_sae_features_pair(self, chosen_text: str, rejected_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        activation_layer = self.layer + 1

        tokens = self.tokenize_pair(chosen_text, rejected_text)
        chosen_ids  = tokens['chosen_input_ids'].to(self.device)
        chosen_mask = tokens['chosen_attention_mask'].to(self.device)
        rejected_ids  = tokens['rejected_input_ids'].to(self.device)
        rejected_mask = tokens['rejected_attention_mask'].to(self.device)

        with torch.no_grad():
            outputs_ch = self.rm_model.model(chosen_ids, attention_mask=chosen_mask, output_hidden_states=True)
            hidden_ch = outputs_ch.hidden_states[activation_layer]
            outputs_rj = self.rm_model.model(rejected_ids, attention_mask=rejected_mask, output_hidden_states=True)
            hidden_rj = outputs_rj.hidden_states[activation_layer]

            if self.use_sae:
                pooled_ch = self.sae.encode(hidden_ch[0, chosen_mask[0].bool(), :]).mean(dim=0)
                pooled_rj = self.sae.encode(hidden_rj[0, rejected_mask[0].bool(), :]).mean(dim=0)
            else:
                pooled_ch = hidden_ch[0, chosen_mask[0].bool(), :].mean(dim=0).float()
                pooled_rj = hidden_rj[0, rejected_mask[0].bool(), :].mean(dim=0).float()

        return pooled_ch, pooled_rj
    
    def get_original_rewards(self, chosen_text: str, rejected_text: str) -> Tuple[float, float]:
        """Get rewards from the base reward model."""
        tokens = self.tokenize_pair(chosen_text, rejected_text)
        chosen_ids = tokens['chosen_input_ids'].to(self.device)
        chosen_mask = tokens['chosen_attention_mask'].to(self.device)
        rejected_ids = tokens['rejected_input_ids'].to(self.device)
        rejected_mask = tokens['rejected_attention_mask'].to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'score_model':
                outputs_ch = self.rm_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                outputs_rj = self.rm_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                ch_reward = outputs_ch.end_scores.squeeze().cpu().item()
                rj_reward = outputs_rj.end_scores.squeeze().cpu().item()
            else:
                outputs_ch = self.rm_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                outputs_rj = self.rm_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                ch_reward = outputs_ch.logits.squeeze().cpu().item()
                rj_reward = outputs_rj.logits.squeeze().cpu().item()
        
        return ch_reward, rj_reward
    
    def train_correction_head(self, train_cases: List[Dict],
                             num_epochs: int = 100, batch_size: int = 32,
                             learning_rate: float = 1e-3,
                             reg_weight: float = 0.01,
                             l1_weight: float = 0.0):
        self.correction_head = ResidualCorrectionHead(
            feature_dim=self.feature_dim,
            use_bias=True
        ).to(self.device)

        data_list = []
        
        for case in tqdm(train_cases, desc="Collecting data"):
            benign_chosen = case.get('original_chosen')
            benign_rejected = case.get('original_rejected')
            adv_chosen = case.get('adv_chosen', case.get('hacked_chosen'))
            adv_rejected = case.get('adv_rejected', case.get('hacked_rejected'))
            
            if not all([benign_chosen, benign_rejected, adv_chosen, adv_rejected]):
                continue
            
            benign_ch_feat, benign_rj_feat = self.compute_sae_features_pair(benign_chosen, benign_rejected)
            benign_ch_reward, benign_rj_reward = self.get_original_rewards(benign_chosen, benign_rejected)
            
            adv_ch_feat, adv_rj_feat = self.compute_sae_features_pair(adv_chosen, adv_rejected)
            adv_ch_reward, adv_rj_reward = self.get_original_rewards(adv_chosen, adv_rejected)
            
            data_list.append({
                'type': 'benign',
                'chosen_features': benign_ch_feat,
                'rejected_features': benign_rj_feat,
                'chosen_original_reward': benign_ch_reward,
                'rejected_original_reward': benign_rj_reward
            })
            
            data_list.append({
                'type': 'adversarial',
                'chosen_features': adv_ch_feat,
                'rejected_features': adv_rj_feat,
                'chosen_original_reward': adv_ch_reward,
                'rejected_original_reward': adv_rj_reward
            })
        
        train_data = data_list
        optimizer = optim.Adam(self.correction_head.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_train_acc = 0
        best_state = None
        
        for epoch in range(num_epochs):
            self.correction_head.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                optimizer.zero_grad()
                
                batch_loss = 0
                for sample in batch:
                    ch_feat = sample['chosen_features'].unsqueeze(0)
                    rj_feat = sample['rejected_features'].unsqueeze(0)
                    ch_orig = sample['chosen_original_reward']
                    rj_orig = sample['rejected_original_reward']
                    
                    ch_correction = self.correction_head(ch_feat).squeeze()
                    rj_correction = self.correction_head(rj_feat).squeeze()
                    
                    ch_corrected = ch_orig + ch_correction
                    rj_corrected = rj_orig + rj_correction
                    
                    if sample['type'] == 'benign':
                        reg_loss = reg_weight * (ch_correction ** 2 + rj_correction ** 2)
                        margin = 1.0
                        ranking_loss = torch.relu(margin - (ch_corrected - rj_corrected))
                        l1_loss = l1_weight * torch.sum(torch.abs(self.correction_head.linear.weight)) if l1_weight > 0 else 0
                        loss = ranking_loss + reg_loss + l1_loss
                    else:
                        margin = 1.0
                        ranking_loss = torch.relu(margin - (ch_corrected - rj_corrected))
                        l1_loss = l1_weight * torch.sum(torch.abs(self.correction_head.linear.weight)) if l1_weight > 0 else 0
                        loss = ranking_loss + l1_loss
                    
                    batch_loss += loss
                    
                    if ch_corrected > rj_corrected:
                        train_correct += 1
                    train_total += 1
                
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.correction_head.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += batch_loss.item()
            
            train_loss /= (len(train_data) / batch_size)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            scheduler.step()
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_state = self.correction_head.state_dict()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.1%}")
        
        if best_state is not None:
            self.correction_head.load_state_dict(best_state)
            print(f"\nBest Train Acc: {best_train_acc:.1%}")
        
        return {'best_acc': best_train_acc}
    
    def predict_with_correction(self, chosen_text: str, rejected_text: str) -> Dict:
        """Score a pair with residual correction applied."""
        if self.correction_head is None:
            raise ValueError("Must train correction head first!")
        
        ch_orig, rj_orig = self.get_original_rewards(chosen_text, rejected_text)
        ch_feat, rj_feat = self.compute_sae_features_pair(chosen_text, rejected_text)
        
        self.correction_head.eval()
        with torch.no_grad():
            ch_correction = self.correction_head(ch_feat.unsqueeze(0)).squeeze().cpu().item()
            rj_correction = self.correction_head(rj_feat.unsqueeze(0)).squeeze().cpu().item()
        
        ch_final = ch_orig + ch_correction
        rj_final = rj_orig + rj_correction
        
        return {
            'chosen_reward': ch_final,
            'rejected_reward': rj_final,
            'defended_diff': ch_final - rj_final,
            'correct': ch_final > rj_final,
            'chosen_correction': ch_correction,
            'rejected_correction': rj_correction,
            'baseline_diff': ch_orig - rj_orig
        }
    
    def evaluate_on_test_set(self, test_cases: List[Dict]) -> Dict:
        """Evaluate on test set."""
        results = {
            'original': {'baseline': 0, 'defended': 0, 'total': 0},
            'adversarial': {'baseline': 0, 'defended': 0, 'total': 0},
            'corrections': [],

            'per_sample': {
                'benign': [],
                'adversarial': []
            }
        }
        
        for idx, case in enumerate(tqdm(test_cases, desc="Evaluating")):
            initial_rewards = case.get('initial_rewards', {})
            correct_diff = initial_rewards.get('diff', None)
            
            # === Original (Benign) samples ===
            if 'original_chosen' in case and 'original_rejected' in case:
                ch_base, rj_base = self.get_original_rewards(
                    case['original_chosen'], case['original_rejected'])
                baseline_diff = ch_base - rj_base
                
                if baseline_diff > 0:
                    results['original']['baseline'] += 1
                
                defended = self.predict_with_correction(
                    case['original_chosen'], case['original_rejected'])
                if defended['correct']:
                    results['original']['defended'] += 1
                
                results['original']['total'] += 1
                
                # Benign: by definition, before-defense DRR = 1.0
                defended_drr = compute_diff_recovery_rate(baseline_diff, defended['defended_diff'])
                
                results['per_sample']['benign'].append({
                    'index': idx,
                    'correct_diff': float(baseline_diff),
                    'before_defense_diff': float(baseline_diff),
                    'after_defense_diff': float(defended['defended_diff']),
                    'before_drr': 1.0,
                    'after_drr': float(defended_drr),
                    'drr_change': float(defended_drr - 1.0),
                    'baseline_correct': baseline_diff > 0,
                    'defended_correct': defended['correct']
                })
            
            # === Adversarial samples ===
            adv_chosen = case.get('adv_chosen', case.get('hacked_chosen'))
            adv_rejected = case.get('adv_rejected', case.get('hacked_rejected'))
            
            if adv_chosen and adv_rejected and correct_diff is not None:
                ch_base, rj_base = self.get_original_rewards(adv_chosen, adv_rejected)
                baseline_diff = ch_base - rj_base
                
                if baseline_diff > 0:
                    results['adversarial']['baseline'] += 1
                
                defended = self.predict_with_correction(adv_chosen, adv_rejected)
                if defended['correct']:
                    results['adversarial']['defended'] += 1
                
                results['adversarial']['total'] += 1
                
                results['corrections'].append({
                    'chosen_correction': defended['chosen_correction'],
                    'rejected_correction': defended['rejected_correction']
                })
                
                before_drr = compute_diff_recovery_rate(correct_diff, baseline_diff)
                after_drr = compute_diff_recovery_rate(correct_diff, defended['defended_diff'])
                
                results['per_sample']['adversarial'].append({
                    'index': idx,
                    'correct_diff': float(correct_diff),
                    'before_defense_diff': float(baseline_diff),
                    'after_defense_diff': float(defended['defended_diff']),
                    'before_drr': float(before_drr),
                    'after_drr': float(after_drr),
                    'drr_change': float(after_drr - before_drr),
                    'baseline_correct': baseline_diff > 0,
                    'defended_correct': defended['correct'],
                    'chosen_correction': float(defended['chosen_correction']),
                    'rejected_correction': float(defended['rejected_correction'])
                })
        
        # Calculate accuracies
        for key in ['original', 'adversarial']:
            if results[key]['total'] > 0:
                results[key]['baseline_acc'] = results[key]['baseline'] / results[key]['total']
                results[key]['defended_acc'] = results[key]['defended'] / results[key]['total']
                results[key]['improvement'] = results[key]['defended_acc'] - results[key]['baseline_acc']
        
        results['drr_summary'] = self._compute_drr_summary(results['per_sample'])
        
        return results
    
    def _compute_drr_summary(self, per_sample: Dict) -> Dict:
        """Compute DRR summary statistics."""
        summary = {}
        
        if per_sample['benign']:
            data = per_sample['benign']
            after_drr = [d['after_drr'] for d in data]
            drr_change = [d['drr_change'] for d in data]
            
            summary['benign'] = {
                'count': len(data),
                'before_drr': {'mean': 1.0, 'std': 0.0},
                'after_drr': {
                    'mean': float(np.mean(after_drr)),
                    'std': float(np.std(after_drr)),
                    'median': float(np.median(after_drr)),
                    'min': float(np.min(after_drr)),
                    'max': float(np.max(after_drr))
                },
                'drr_change': {
                    'mean': float(np.mean(drr_change)),
                    'std': float(np.std(drr_change))
                },
                'distribution': {
                    'perfect (=1.0)': sum(1 for d in after_drr if abs(d - 1.0) < 0.01) / len(after_drr),
                    'over (>1.0)': sum(1 for d in after_drr if d > 1.0) / len(after_drr),
                    'good (0.8-1.0)': sum(1 for d in after_drr if 0.8 <= d <= 1.0) / len(after_drr),
                    'partial (0-0.8)': sum(1 for d in after_drr if 0 < d < 0.8) / len(after_drr),
                    'wrong (<0)': sum(1 for d in after_drr if d < 0) / len(after_drr)
                }
            }
        
        if per_sample['adversarial']:
            data = per_sample['adversarial']
            before_drr = [d['before_drr'] for d in data]
            after_drr = [d['after_drr'] for d in data]
            drr_change = [d['drr_change'] for d in data]
            
            summary['adversarial'] = {
                'count': len(data),
                'before_drr': {
                    'mean': float(np.mean(before_drr)),
                    'std': float(np.std(before_drr)),
                    'median': float(np.median(before_drr)),
                    'min': float(np.min(before_drr)),
                    'max': float(np.max(before_drr))
                },
                'after_drr': {
                    'mean': float(np.mean(after_drr)),
                    'std': float(np.std(after_drr)),
                    'median': float(np.median(after_drr)),
                    'min': float(np.min(after_drr)),
                    'max': float(np.max(after_drr))
                },
                'drr_change': {
                    'mean': float(np.mean(drr_change)),
                    'std': float(np.std(drr_change)),
                    'median': float(np.median(drr_change)),
                    'min': float(np.min(drr_change)),
                    'max': float(np.max(drr_change))
                },
                'before_distribution': {
                    'correct (>0)': sum(1 for d in before_drr if d > 0) / len(before_drr),
                    'wrong (<=0)': sum(1 for d in before_drr if d <= 0) / len(before_drr)
                },
                'after_distribution': {
                    'over (>1.0)': sum(1 for d in after_drr if d > 1.0) / len(after_drr),
                    'good (0.8-1.0)': sum(1 for d in after_drr if 0.8 <= d <= 1.0) / len(after_drr),
                    'partial (0-0.8)': sum(1 for d in after_drr if 0 < d < 0.8) / len(after_drr),
                    'wrong (<=0)': sum(1 for d in after_drr if d <= 0) / len(after_drr)
                }
            }
        
        return summary
    
    def save_model(self, save_dir: str):
        """Save model."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.correction_head is not None:
            torch.save(self.correction_head.state_dict(), os.path.join(save_dir, 'correction_head.pth'))
        
        config = {'feature_dim': self.feature_dim, 'layer': self.layer}
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, load_dir: str):
        self.correction_head = ResidualCorrectionHead(
            feature_dim=self.feature_dim, use_bias=True
        ).to(self.device)
        self.correction_head.load_state_dict(
            torch.load(os.path.join(load_dir, 'correction_head.pth'), map_location=self.device))


def filter_successfully_attacked_cases(cases: List[Dict], threshold: float = 0.0) -> List[Dict]:
    return [c for c in cases if c['initial_rewards']['diff'] > threshold 
            and c['final_rewards']['diff'] <= -threshold]


def get_short_name(path: str) -> str:
    """Extract a short identifier from a path for use in filenames."""
    name = os.path.basename(path.rstrip('/'))
    name = os.path.splitext(name)[0]
    name = name.replace('-', '_').replace('.', '_')
    return name


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Residual Correction Defense')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--feature_type', type=str, default='sae', choices=['sae', 'raw'],
                       help='Feature type: sae (SAE-encoded) or raw (hidden state)')
    parser.add_argument('--sae_path', type=str, default=None,
                       help='SAE checkpoint (required when --feature_type sae)')
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                       help='One or more dataset JSON files; cases are pooled before split')
    parser.add_argument('--output_dir', type=str, default='./residual_defense')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--max_length', type=int, default=512)
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--reg_weight', type=float, default=0.05)
    parser.add_argument('--l1_weight', type=float, default=0.0)
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--mode', type=str, default='train_and_test',
                       choices=['train_and_test', 'train_only', 'test_only'])
    parser.add_argument('--load_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_type = detect_model_type(args.model_name)
    use_chat_template = detect_chat_template_model(args.model_name)

    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    rm_model, _ = load_model(args.model_name, args.cache_dir, model_type)

    use_sae = (args.feature_type == 'sae')
    sae = None
    if use_sae:
        if not args.sae_path:
            raise ValueError("--sae_path required when --feature_type sae")
        sae = SAE.load_from_pretrained(args.sae_path, device="cpu").to(device)
        feature_dim = sae.cfg.d_sae
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        feature_dim = cfg.hidden_size

    defense_system = ResidualCorrectionDefense(
        rm_model=rm_model, tokenizer=tokenizer, sae=sae,
        layer=args.layer, model_type=model_type, feature_dim=feature_dim,
        max_length=args.max_length, use_chat_template=use_chat_template,
        use_sae=use_sae
    )
    
    all_cases = []
    for path in args.dataset_paths:
        with open(path, 'r') as f:
            cases = json.load(f)
        all_cases.extend(cases)
    print(f"Loaded {len(all_cases)} cases")

    attacked_cases = filter_successfully_attacked_cases(all_cases)
    n_train = int(len(attacked_cases) * args.train_ratio)
    train_cases = attacked_cases[:n_train]
    test_cases = attacked_cases[n_train:]
    print(f"Perturbation cases: {len(attacked_cases)} ({len(train_cases)} train / {len(test_cases)} test)")

    model_short = get_short_name(args.model_name)
    dataset_short = f"multi{len(args.dataset_paths)}" if len(args.dataset_paths) > 1 else get_short_name(args.dataset_paths[0])
    epoch_str = f"ep{args.num_epochs}"
    
    if args.mode in ['train_and_test', 'train_only']:
        defense_system.train_correction_head(
            train_cases, num_epochs=args.num_epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            reg_weight=args.reg_weight, l1_weight=args.l1_weight
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        defense_system.save_model(args.output_dir)
    
    if args.mode in ['train_and_test', 'test_only']:
        if args.mode == 'test_only':
            if not args.load_dir:
                raise ValueError("Must specify --load_dir")
            defense_system.load_model(args.load_dir)

        results = defense_system.evaluate_on_test_set(test_cases)

        print("\nBenign:")
        if results['original']['total'] > 0:
            print(f"  Baseline:  {results['original']['baseline_acc']:.1%}")
            print(f"  Defended:  {results['original']['defended_acc']:.1%}")
            print(f"  Change:    {results['original']['improvement']:+.1%}")
        
        print("\nAdversarial:")
        if results['adversarial']['total'] > 0:
            print(f"  Baseline:  {results['adversarial']['baseline_acc']:.1%}")
            print(f"  Defended:  {results['adversarial']['defended_acc']:.1%}")
            print(f"  Change:    {results['adversarial']['improvement']:+.1%}")

        results_filename = f'results_{model_short}_{dataset_short}_{epoch_str}.json'
        with open(os.path.join(args.output_dir, results_filename), 'w') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                return obj
            
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nResults saved to: {os.path.join(args.output_dir, results_filename)}")
        
        per_sample_filename = f'per_sample_diff_{model_short}_{dataset_short}_{epoch_str}.json'
        per_sample_path = os.path.join(args.output_dir, per_sample_filename)
        with open(per_sample_path, 'w') as f:
            json.dump(results['per_sample'], f, indent=2)
        print(f"Per-sample diff data saved to: {per_sample_path}")
    
    print(f"\n✅ Done! Results in {args.output_dir}")


if __name__ == "__main__":
    main()