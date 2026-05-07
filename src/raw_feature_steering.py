import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlamaModel, LlamaPreTrainedModel, LlamaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
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

def tokenize_single(text: str, tokenizer, max_length: int = 512, use_chat_template: bool = False):
    """Tokenize a single text"""
    if use_chat_template:
        formatted = format_with_chat_template(text, tokenizer)
        tokens = tokenizer(formatted, add_special_tokens=True, truncation=True, max_length=max_length, padding=False, return_tensors='pt')
        input_ids = tokens['input_ids'][0]
        attention_mask = tokens['attention_mask'][0]
    else:
        if "Human:" in text and "Assistant:" in text:
            prompt, answer = process_dialogue_to_prompt(text)
            full_text = prompt + answer
        else:
            full_text = text
        input_ids = tokenizer(full_text, add_special_tokens=True, truncation=False, return_tensors='pt')['input_ids'][0]
        max_length = min(max_length, tokenizer.model_max_length)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    return {
        'input_ids': input_ids.unsqueeze(0),
        'attention_mask': attention_mask.unsqueeze(0)
    }

def tokenize_pair(chosen: str, rejected: str, tokenizer, max_length: int = 512, use_chat_template: bool = False):
    """Tokenize a pair for evaluation"""
    chosen_tokens = tokenize_single(chosen, tokenizer, max_length, use_chat_template)
    rejected_tokens = tokenize_single(rejected, tokenizer, max_length, use_chat_template)
    
    chosen_ids = chosen_tokens['input_ids'][0]
    chosen_mask = chosen_tokens['attention_mask'][0]
    rejected_ids = rejected_tokens['input_ids'][0]
    rejected_mask = rejected_tokens['attention_mask'][0]
    
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


# --- Diff Recovery Rate (DRR) ---
def compute_diff_recovery_rate(correct_diff: float, current_diff: float, eps: float = 1e-6) -> float:
    if abs(correct_diff) <= eps:
        return 0.0
    return current_diff / correct_diff


# --- Hidden State Feature Steering Defense (Individual Samples) ---
class HiddenStateSteeringDefense:
    """
    Feature steering defense operating on raw hidden states.
    Treats chosen and rejected as individual samples rather than pairs.
    Computes a steering vector as mean(adv) - mean(benign) and subtracts it at inference.
    """
    
    def __init__(self, rm_model, tokenizer, layer: int, model_type: str,
                 hidden_dim: int, max_length: int = 512, use_chat_template: bool = False):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.layer = layer
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.device = device
        
        # Steering vector
        self.steering_vector = None
        self.adv_hidden_states = []
        self.benign_hidden_states = []
    
    def tokenize_single(self, text: str) -> Dict:
        return tokenize_single(text, self.tokenizer, self.max_length, self.use_chat_template)
    
    def tokenize_pair(self, chosen: str, rejected: str) -> Dict:
        return tokenize_pair(chosen, rejected, self.tokenizer, self.max_length, self.use_chat_template)
    
    def extract_hidden_states_single(self, text: str) -> torch.Tensor:
        """Extract hidden states for a single text with mean pooling."""
        activation_layer = self.layer + 1
        
        tokens = self.tokenize_single(text)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.rm_model.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[activation_layer]
            non_pad_mask = attention_mask[0].bool()
            pooled = hidden[0, non_pad_mask, :].mean(dim=0)
        
        return pooled
    
    def train_steering_vector(self, train_cases: List[Dict], steering_strength: float = 1.0):
        self.adv_hidden_states = []
        self.benign_hidden_states = []
        
        for case in tqdm(train_cases, desc="Collecting hidden states"):
            adv_chosen = case.get('adv_chosen', case.get('hacked_chosen'))
            adv_rejected = case.get('adv_rejected', case.get('hacked_rejected'))
            benign_chosen = case.get('original_chosen')
            benign_rejected = case.get('original_rejected')
            
            # Add adversarial samples individually
            if adv_chosen:
                adv_hidden = self.extract_hidden_states_single(adv_chosen)
                self.adv_hidden_states.append(adv_hidden.cpu())
            
            if adv_rejected:
                adv_hidden = self.extract_hidden_states_single(adv_rejected)
                self.adv_hidden_states.append(adv_hidden.cpu())
            
            # Add benign samples individually
            if benign_chosen:
                benign_hidden = self.extract_hidden_states_single(benign_chosen)
                self.benign_hidden_states.append(benign_hidden.cpu())
            
            if benign_rejected:
                benign_hidden = self.extract_hidden_states_single(benign_rejected)
                self.benign_hidden_states.append(benign_hidden.cpu())
        
        adv_mean = torch.stack(self.adv_hidden_states).mean(dim=0)
        benign_mean = torch.stack(self.benign_hidden_states).mean(dim=0)
        
        self.steering_vector = (adv_mean - benign_mean) * steering_strength
        self.steering_vector = self.steering_vector.to(self.device)
        return self.steering_vector
    
    def apply_steering_and_evaluate(self, chosen_text: str, rejected_text: str) -> Dict:
        """Apply steering vector and evaluate."""
        if self.steering_vector is None:
            raise ValueError("Must train steering vector first!")
        
        activation_layer = self.layer + 1
        
        tokens = self.tokenize_pair(chosen_text, rejected_text)
        input_ids = torch.cat([tokens['chosen_input_ids'], tokens['rejected_input_ids']], dim=0).to(self.device)
        attention_mask = torch.cat([tokens['chosen_attention_mask'], tokens['rejected_attention_mask']], dim=0).to(self.device)
        
        non_pad_masks = [attention_mask[0].bool(), attention_mask[1].bool()]
        
        def create_hook(masks, steering_vec):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                modified = hidden.clone()
                
                with torch.no_grad():
                    for i in range(2):
                        mask = masks[i]
                        modified[i, mask, :] = modified[i, mask, :] - steering_vec.unsqueeze(0)
                
                return (modified, *output[1:]) if isinstance(output, tuple) else modified
            return hook_fn
        
        hook = self.rm_model.model.layers[self.layer].register_forward_hook(
            create_hook(non_pad_masks, self.steering_vector))
        
        try:
            with torch.no_grad():
                if self.model_type == 'score_model':
                    outputs = self.rm_model(input_ids=input_ids, attention_mask=attention_mask)
                    chosen_score = outputs.end_scores[0].cpu().item()
                    rejected_score = outputs.end_scores[1].cpu().item()
                else:
                    outputs = self.rm_model(input_ids=input_ids, attention_mask=attention_mask)
                    chosen_score = outputs.logits[0].cpu().item()
                    rejected_score = outputs.logits[1].cpu().item()
                
                diff = chosen_score - rejected_score
        finally:
            hook.remove()
        
        return {
            'chosen_score': chosen_score,
            'rejected_score': rejected_score,
            'defended_diff': diff,
            'correct': diff > 0
        }
    
    def evaluate_without_steering(self, chosen_text: str, rejected_text: str) -> Dict:
        """Baseline evaluation without steering."""
        tokens = self.tokenize_pair(chosen_text, rejected_text)
        chosen_ids = tokens['chosen_input_ids'].to(self.device)
        chosen_mask = tokens['chosen_attention_mask'].to(self.device)
        rejected_ids = tokens['rejected_input_ids'].to(self.device)
        rejected_mask = tokens['rejected_attention_mask'].to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'score_model':
                outputs_ch = self.rm_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                outputs_rj = self.rm_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                chosen_score = outputs_ch.end_scores.squeeze().cpu().item()
                rejected_score = outputs_rj.end_scores.squeeze().cpu().item()
            else:
                outputs_ch = self.rm_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                outputs_rj = self.rm_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                chosen_score = outputs_ch.logits.squeeze().cpu().item()
                rejected_score = outputs_rj.logits.squeeze().cpu().item()
            
            diff = chosen_score - rejected_score
        
        return {
            'chosen_score': chosen_score,
            'rejected_score': rejected_score,
            'baseline_diff': diff,
            'correct': diff > 0
        }
    
    def evaluate_on_test_set(self, test_cases: List[Dict]) -> Dict:
        """Evaluate on test set."""
        results = {
            'original': {'baseline': 0, 'defended': 0, 'total': 0},
            'adversarial': {'baseline': 0, 'defended': 0, 'total': 0},
            'per_sample': {
                'benign': [],
                'adversarial': []
            }
        }
        
        for idx, case in enumerate(tqdm(test_cases, desc="Evaluating")):
            initial_rewards = case.get('initial_rewards', {})
            correct_diff = initial_rewards.get('diff', None)
            
            # === Benign samples ===
            if 'original_chosen' in case and 'original_rejected' in case:
                baseline = self.evaluate_without_steering(
                    case['original_chosen'], case['original_rejected'])
                baseline_diff = baseline['baseline_diff']
                
                if baseline_diff > 0:
                    results['original']['baseline'] += 1
                
                defended = self.apply_steering_and_evaluate(
                    case['original_chosen'], case['original_rejected'])
                if defended['correct']:
                    results['original']['defended'] += 1
                
                results['original']['total'] += 1
                
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
                baseline = self.evaluate_without_steering(adv_chosen, adv_rejected)
                baseline_diff = baseline['baseline_diff']
                
                if baseline_diff > 0:
                    results['adversarial']['baseline'] += 1
                
                defended = self.apply_steering_and_evaluate(adv_chosen, adv_rejected)
                if defended['correct']:
                    results['adversarial']['defended'] += 1
                
                results['adversarial']['total'] += 1
                
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
                    'defended_correct': defended['correct']
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
                'drr_change': {'mean': float(np.mean(drr_change)), 'std': float(np.std(drr_change))},
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
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.steering_vector.cpu(), os.path.join(save_dir, 'steering_vector.pt'))
        
        config = {
            'hidden_dim': self.hidden_dim,
            'layer': self.layer,
            'model_type': self.model_type,
            'vector_norm': float(self.steering_vector.norm().item()),
            'n_adv_samples': len(self.adv_hidden_states),
            'n_benign_samples': len(self.benign_hidden_states),
            'mode': 'individual_samples'  # Mark that this uses individual samples
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, load_dir: str):
        self.steering_vector = torch.load(os.path.join(load_dir, 'steering_vector.pt')).to(self.device)


def filter_successfully_attacked_cases(cases: List[Dict], threshold: float = 0.0) -> List[Dict]:
    return [c for c in cases if c['initial_rewards']['diff'] > threshold 
            and c['final_rewards']['diff'] <= -threshold]


def print_drr_summary(results: Dict):
    """Print DRR summary."""
    summary = results.get('drr_summary', {})
    
    print("\n" + "="*70)
    print("    DIFF RECOVERY RATE (DRR) SUMMARY")
    print("="*70)
    
    if 'adversarial' in summary:
        adv = summary['adversarial']
        print("\n📊 Adversarial Samples:")
        print(f"  Count: {adv['count']}")
        print(f"\n  ┌─────────────────┬──────────────┬──────────────┐")
        print(f"  │                 │ Before Def.  │ After Def.   │")
        print(f"  ├─────────────────┼──────────────┼──────────────┤")
        print(f"  │ DRR Mean        │ {adv['before_drr']['mean']:>10.4f}   │ {adv['after_drr']['mean']:>10.4f}   │")
        print(f"  │ DRR Median      │ {adv['before_drr']['median']:>10.4f}   │ {adv['after_drr']['median']:>10.4f}   │")
        print(f"  └─────────────────┴──────────────┴──────────────┘")
        print(f"\n  DRR Improvement: Mean={adv['drr_change']['mean']:+.4f}")
    
    if 'benign' in summary:
        benign = summary['benign']
        print("\n📊 Benign Samples:")
        print(f"  Count: {benign['count']}")
        print(f"  After Defense DRR: Mean={benign['after_drr']['mean']:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hidden State Feature Steering Defense (Individual Samples)')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                       help='One or more dataset JSON files; cases are pooled before split')
    parser.add_argument('--output_dir', type=str, default='./hidden_steering_defense_individual')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--steering_strength', type=float, default=1.0)
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
    
    if hasattr(rm_model.config, 'hidden_size'):
        hidden_dim = rm_model.config.hidden_size
    else:
        hidden_dim = rm_model.model.config.hidden_size
    
    defense_system = HiddenStateSteeringDefense(
        rm_model=rm_model, tokenizer=tokenizer,
        layer=args.layer, model_type=model_type, hidden_dim=hidden_dim,
        max_length=args.max_length, use_chat_template=use_chat_template
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
    
    if args.mode in ['train_and_test', 'train_only']:
        defense_system.train_steering_vector(train_cases, steering_strength=args.steering_strength)
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
        
        print_drr_summary(results)
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(args.output_dir, 'per_sample_diff.json'), 'w') as f:
            json.dump(results['per_sample'], f, indent=2)
        print(f"\nPer-sample diff data saved to: {args.output_dir}/per_sample_diff.json")
    
    print(f"\n✅ Done! Results in {args.output_dir}")


if __name__ == "__main__":
    main()