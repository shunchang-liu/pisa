import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig, LlamaModel, LlamaPreTrainedModel, LlamaTokenizer
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from sae_lens import SAE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from tqdm import tqdm
import pickle
from datetime import datetime
from typing import Tuple, Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Output Definition ---
@dataclass
class ScoreModelOutput(ModelOutput):
    scores: torch.FloatTensor = None
    end_scores: torch.FloatTensor = None

# --- LlamaForScore (Minimal) ---
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

# --- Simple Binary Classifier using Cosine Similarity ---
class SAEClassifier(nn.Module):
    """
    Binary classifier using cosine similarity and absolute difference of SAE features.
    """
    def __init__(self, sae_dim, hidden_dims=[128, 64], dropout=0.3):
        super(SAEClassifier, self).__init__()

        # Input: absolute difference features + cosine similarity
        # input_dim = sae_dim + 1
        input_dim = sae_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Binary classification
        self.model = nn.Sequential(*layers)
    
    def compute_cosine_similarity(self, x1, x2):
        """Compute cosine similarity between two vectors (batch-wise)"""
        x1_norm = torch.norm(x1, dim=1, keepdim=True)
        x2_norm = torch.norm(x2, dim=1, keepdim=True)
        
        eps = 1e-8
        x1_normalized = x1 / (x1_norm + eps)
        x2_normalized = x2 / (x2_norm + eps)
        
        cosine_sim = torch.sum(x1_normalized * x2_normalized, dim=1, keepdim=True)
        
        return cosine_sim
    
    def forward(self, features1, features2):
        abs_diff = torch.abs(features1 - features2)
        expected_dim = self.model[0].in_features
        if abs_diff.shape[1] < expected_dim:
            # legacy checkpoint trained with cosine similarity appended
            cosine_sim = self.compute_cosine_similarity(features1, features2)
            x = torch.cat([abs_diff, cosine_sim], dim=1)
        else:
            x = abs_diff
        return self.model(x)

# --- Model Type Detection ---
def detect_model_type(model_name: str) -> str:
    """Detect the type of model based on its name."""
    model_name_lower = model_name.lower()
    
    # Chat template models
    chat_template_models = ['skywork', 'reward-v2', 'chat', 'instruct']
    if any(template_key in model_name_lower for template_key in chat_template_models):
        return 'chat_template'
    
    # Score models (typically Llama-based reward models)
    if 'poisoned' in model_name_lower or 'beaver' in model_name_lower or 'rm' in model_name_lower:
        return 'score_model'
    
    # Default to standard sequence classification
    return 'standard'

def detect_chat_template_model(model_name: str) -> bool:
    """Detect if the model uses chat templates."""
    chat_template_models = ['skywork', 'reward-v2', 'chat', 'instruct']
    model_name_lower = model_name.lower()
    return any(template_key in model_name_lower for template_key in chat_template_models)

# --- Tokenizer Loading ---
def load_tokenizer(model_name: str, cache_dir: str):
    """Load the appropriate tokenizer for the model."""
    model_name_lower = model_name.lower()
    
    # Check if this is a chat template model
    use_chat_template = detect_chat_template_model(model_name)
    
    # List of known Llama-based models
    llama_based_models = [
        'poisoned-rlhf', 'poisoned-reward', 'alpaca', 'vicuna', 'koala',
        'wizardlm', 'guanaco'
    ]
    
    # Check if we should use LlamaTokenizer
    use_llama_tokenizer = False
    
    # Skip LlamaTokenizer for chat template models
    if not use_chat_template:
        # Check for 'llama' in name or known Llama-based models
        if 'llama' in model_name_lower:
            use_llama_tokenizer = True
        else:
            for llama_variant in llama_based_models:
                if llama_variant in model_name_lower:
                    use_llama_tokenizer = True
                    break
        
        # Also check model config if available
        if not use_llama_tokenizer:
            try:
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
                if hasattr(config, 'model_type') and 'llama' in config.model_type.lower():
                    use_llama_tokenizer = True
            except:
                pass
    
    # Load appropriate tokenizer
    if use_llama_tokenizer:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

# --- Model Loading ---
def load_model(model_name: str, cache_dir: str, model_type: str = None):
    """Load the appropriate model based on its type."""
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    if model_type == 'score_model':
        model = LlamaForScore.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto" if device.type == "cuda" else None
        )
    elif model_type == 'chat_template':
        from transformers import AutoModelForSequenceClassification
        
        # Determine dtype based on model
        if 'skywork' in model_name.lower():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device.type == "cuda" else None,
                attn_implementation="flash_attention_2",
                num_labels=1,
                cache_dir=cache_dir
            )
        except:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device.type == "cuda" else None,
                num_labels=1,
                cache_dir=cache_dir
            )
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            cache_dir=cache_dir,
            device_map="auto" if device.type == "cuda" else None
        )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    return model, model_type

# --- Case Filtering Functions ---
def filter_successfully_attacked_cases(cases, threshold=0.0):
    """Filter cases where attack succeeded (flipped from positive to negative)"""
    return [c for c in cases if c['initial_rewards']['diff'] > threshold 
            and c['final_rewards']['diff'] <= -threshold]

# --- Text Processing Utils (ALIGNED WITH CODE 1) ---
def format_with_chat_template(text: str, tokenizer) -> str:
    """Format text using chat template for models like Skywork."""
    # Parse the dialogue format into chat format
    if "Human:" in text and "Assistant:" in text:
        # Split dialogue and process
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
        
        # Add the last message
        if current_role and current_content:
            messages.append({"role": current_role, "content": " ".join(current_content)})
    else:
        # Assume it's a single turn conversation
        messages = [{"role": "user", "content": text}]
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Remove duplicate BOS token if present
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    
    return formatted

def process_dialogue_to_prompt(dialogue_text: str) -> Tuple[str, str]:
    """Process dialogue text to extract prompt and answer, matching official format."""
    # Split dialogue and process like official code
    split_text = [i for i in dialogue_text.split("\n\n") if i != ""]
    
    dialog = []
    for i, line in enumerate(split_text):
        if line.startswith("Human: "):
            dialog.append(line[7:])  # len('Human: ') == 7
        elif line.startswith("Assistant: "):
            dialog.append(line[11:])  # len('Assistant: ') == 11
        else:
            if len(dialog):
                dialog[-1] += "\n" + line
    
    # Build prompt with USER/ASSISTANT format (matching official PROMPT_USER and PROMPT_ASSISTANT)
    prompt = ""
    for i, line in enumerate(dialog[:-1]):  # All except last answer
        if i % 2 == 0:
            # User input - PROMPT_USER: str = " USER: {input} "
            prompt += f" USER: {line} "
            # Add PROMPT_ASSISTANT: str = "ASSISTANT:"
            prompt += "ASSISTANT: "
        else:
            # Assistant response (no extra formatting)
            prompt += f"{line}"
    
    # Last item is the answer
    answer = dialog[-1] if dialog else ""
    
    return prompt, answer

def tokenize_pair(chosen: str, rejected: str, tokenizer, max_length: int = 512, 
                  use_chat_template: bool = False):
    """Tokenize a pair of texts following the official PreferenceDataset approach."""
    
    if use_chat_template:
        # Format with chat template
        chosen_formatted = format_with_chat_template(chosen, tokenizer)
        rejected_formatted = format_with_chat_template(rejected, tokenizer)
        
        # Tokenize
        chosen_tokens = tokenizer(
            chosen_formatted,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt'
        )
        rejected_tokens = tokenizer(
            rejected_formatted,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt'
        )
        
        chosen_ids = chosen_tokens['input_ids'][0]
        rejected_ids = rejected_tokens['input_ids'][0]
        chosen_mask = chosen_tokens['attention_mask'][0]
        rejected_mask = rejected_tokens['attention_mask'][0]
    else:
        # Original processing
        # Check if texts are in dialogue format (contain Human:/Assistant:)
        if "Human:" in chosen and "Assistant:" in chosen:
            # Process dialogue format
            chosen_prompt, chosen_answer = process_dialogue_to_prompt(chosen)
            rejected_prompt, rejected_answer = process_dialogue_to_prompt(rejected)
            
            # Combine prompt and answer
            chosen_full = chosen_prompt + chosen_answer
            rejected_full = rejected_prompt + rejected_answer
        else:
            # Already in the right format
            chosen_full = chosen
            rejected_full = rejected
        
        # Tokenize without padding first (like official code)
        chosen_ids = tokenizer(
            chosen_full,
            add_special_tokens=True,
            truncation=False,  # Don't truncate yet
            return_tensors='pt'
        )['input_ids'][0]
        
        rejected_ids = tokenizer(
            rejected_full,
            add_special_tokens=True,
            truncation=False,  # Don't truncate yet
            return_tensors='pt'
        )['input_ids'][0]
        
        # Take the last max_length tokens (like official: input_ids[-self.tokenizer.model_max_length:])
        max_length = min(max_length, tokenizer.model_max_length)
        if len(chosen_ids) > max_length:
            chosen_ids = chosen_ids[-max_length:]
        if len(rejected_ids) > max_length:
            rejected_ids = rejected_ids[-max_length:]
        
        # Create attention masks
        chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool)
        rejected_mask = torch.ones_like(rejected_ids, dtype=torch.bool)
    
    # Apply right padding to match lengths
    max_len = max(len(chosen_ids), len(rejected_ids))
    pad_token_id = tokenizer.pad_token_id
    
    # Right pad chosen
    if len(chosen_ids) < max_len:
        pad_len = max_len - len(chosen_ids)
        chosen_ids = torch.cat([
            chosen_ids,
            torch.full((pad_len,), pad_token_id, dtype=chosen_ids.dtype)
        ])
        chosen_mask = torch.cat([
            chosen_mask,
            torch.zeros(pad_len, dtype=torch.bool)
        ])
    
    # Right pad rejected
    if len(rejected_ids) < max_len:
        pad_len = max_len - len(rejected_ids)
        rejected_ids = torch.cat([
            rejected_ids,
            torch.full((pad_len,), pad_token_id, dtype=rejected_ids.dtype)
        ])
        rejected_mask = torch.cat([
            rejected_mask,
            torch.zeros(pad_len, dtype=torch.bool)
        ])
    
    return {
        'chosen_input_ids': chosen_ids.unsqueeze(0),
        'chosen_attention_mask': chosen_mask.unsqueeze(0),
        'rejected_input_ids': rejected_ids.unsqueeze(0),
        'rejected_attention_mask': rejected_mask.unsqueeze(0)
    }

# --- Compute Features ---
def compute_raw_features_pair(chosen_text, rejected_text, model, tokenizer, layer, max_length=512, use_chat_template=False):
    """Compute raw hidden state features for both chosen and rejected texts"""
    activation_layer = layer + 1
    
    tokens = tokenize_pair(chosen_text, rejected_text, tokenizer, max_length, use_chat_template)
    
    chosen_input_ids = tokens['chosen_input_ids'].to(device)
    chosen_attention_mask = tokens['chosen_attention_mask'].to(device)
    rejected_input_ids = tokens['rejected_input_ids'].to(device)
    rejected_attention_mask = tokens['rejected_attention_mask'].to(device)
    
    with torch.no_grad():
        chosen_outputs = model.model(chosen_input_ids, attention_mask=chosen_attention_mask, output_hidden_states=True)
        chosen_hidden_states = chosen_outputs.hidden_states
        chosen_acts = chosen_hidden_states[activation_layer]  # Shape: [batch, seq_len, hidden_dim]
        
        chosen_non_pad_mask = chosen_attention_mask[0].bool()  # [seq_len]
        chosen_valid_acts = chosen_acts[0, chosen_non_pad_mask, :]  # [valid_seq_len, hidden_dim]
        chosen_pooled = chosen_valid_acts.mean(dim=0)  # Average pooling over valid sequence only
        
        rejected_outputs = model.model(rejected_input_ids, attention_mask=rejected_attention_mask, output_hidden_states=True)
        rejected_hidden_states = rejected_outputs.hidden_states
        rejected_acts = rejected_hidden_states[activation_layer]  # Shape: [batch, seq_len, hidden_dim]
        
        rejected_non_pad_mask = rejected_attention_mask[0].bool()  # [seq_len]
        rejected_valid_acts = rejected_acts[0, rejected_non_pad_mask, :]  # [valid_seq_len, hidden_dim]
        rejected_pooled = rejected_valid_acts.mean(dim=0)  # Average pooling over valid sequence only
    
    return chosen_pooled.float().cpu().numpy(), rejected_pooled.float().cpu().numpy()

def compute_sae_features_pair(chosen_text, rejected_text, model, tokenizer, sae, layer, max_length=512, use_chat_template=False):
    """Compute SAE features for both chosen and rejected texts"""
    activation_layer = layer + 1
    
    tokens = tokenize_pair(chosen_text, rejected_text, tokenizer, max_length, use_chat_template)
    
    chosen_input_ids = tokens['chosen_input_ids'].to(device)
    chosen_attention_mask = tokens['chosen_attention_mask'].to(device)
    rejected_input_ids = tokens['rejected_input_ids'].to(device)
    rejected_attention_mask = tokens['rejected_attention_mask'].to(device)
    
    with torch.no_grad():
        chosen_outputs = model.model(chosen_input_ids, attention_mask=chosen_attention_mask, output_hidden_states=True)
        chosen_hidden_states = chosen_outputs.hidden_states
        chosen_acts = chosen_hidden_states[activation_layer]  # Shape: [batch, seq_len, hidden_dim]
        
        chosen_non_pad_mask = chosen_attention_mask[0].bool()  # [seq_len]
        chosen_valid_acts = chosen_acts[0, chosen_non_pad_mask, :]  # [valid_seq_len, hidden_dim]
        chosen_sae_features = sae.encode(chosen_valid_acts)  # [valid_seq_len, sae_dim]
        chosen_pooled = chosen_sae_features.mean(dim=0)  # Average pooling over valid sequence only
        
        rejected_outputs = model.model(rejected_input_ids, attention_mask=rejected_attention_mask, output_hidden_states=True)
        rejected_hidden_states = rejected_outputs.hidden_states
        rejected_acts = rejected_hidden_states[activation_layer]  # Shape: [batch, seq_len, hidden_dim]
        
        rejected_non_pad_mask = rejected_attention_mask[0].bool()  # [seq_len]
        rejected_valid_acts = rejected_acts[0, rejected_non_pad_mask, :]  # [valid_seq_len, hidden_dim]
        rejected_sae_features = sae.encode(rejected_valid_acts)  # [valid_seq_len, sae_dim]
        rejected_pooled = rejected_sae_features.mean(dim=0)  # Average pooling over valid sequence only
    
    return chosen_pooled.cpu().numpy(), rejected_pooled.cpu().numpy()

def extract_pair_features(chosen_text, rejected_text, model, tokenizer, sae, layer, 
                          max_length=512, use_sae=True, use_chat_template=False):
    """Extract features for a chosen-rejected pair"""
    if use_sae:
        chosen_features, rejected_features = compute_sae_features_pair(
            chosen_text, rejected_text, model, tokenizer, sae, layer, 
            max_length, use_chat_template
        )
    else:
        chosen_features, rejected_features = compute_raw_features_pair(
            chosen_text, rejected_text, model, tokenizer, layer, 
            max_length, use_chat_template
        )
    return chosen_features, rejected_features

# --- Load Saved Classifier ---
def load_saved_classifier(model_dir: str, device=None):
    """Load a saved classifier and its configuration"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load weights first to infer actual input_dim (handles legacy checkpoints
    # that were saved with input_dim = sae_dim + 1 due to cosine similarity)
    model_path = os.path.join(model_dir, "classifier.pth")
    state_dict = torch.load(model_path, map_location=device)
    actual_input_dim = state_dict["model.0.weight"].shape[1]

    classifier = SAEClassifier(
        sae_dim=actual_input_dim,
        hidden_dims=config.get('hidden_dims', [128, 64]),
        dropout=config.get('dropout', 0.3)
    ).to(device)

    classifier.load_state_dict(state_dict)
    classifier.eval()

    # Load scalers if saved (legacy checkpoints trained with StandardScaler)
    scalers = None
    scalers_path = os.path.join(model_dir, "scalers.pkl")
    if os.path.exists(scalers_path):
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

    return classifier, config, scalers

# --- Evaluate Pre-trained Classifier ---
def evaluate_pretrained_classifier(args, all_cases, rm_model, tokenizer, sae=None, use_chat_template=False):
    """Evaluate a pre-trained classifier on a dataset"""
    
    print("\n" + "="*70)
    print("    EVALUATING PRE-TRAINED CLASSIFIER")
    print("="*70)
    
    # Load pre-trained classifier
    print(f"Loading pre-trained classifier from: {args.pretrained_path}")
    classifier, config, scalers = load_saved_classifier(args.pretrained_path)

    # Extract configuration
    use_sae = config['use_sae']
    feature_type = "SAE" if use_sae else "Raw"
    layer = config['layer']

    print(f"Classifier details:")
    print(f"  - Feature type: {feature_type}")
    print(f"  - Feature dimension: {config['feature_dim']}")
    print(f"  - Layer: {layer}")
    print(f"  - Case type: {config['case_type']}")
    print(f"  - Original model: {config['model_name']}")
    print(f"  - Using chat template: {use_chat_template}")
    print(f"  - Scalers: {'loaded' if scalers else 'none'}")
    
    # Check if SAE is needed
    if use_sae and sae is None:
        if config.get('sae_path'):
            print(f"Loading SAE from config: {config['sae_path']}")
            sae = SAE.load_from_pretrained(config['sae_path'], device="cpu").to(device)
        else:
            raise ValueError("SAE is required but not provided and not found in config!")
    
    # Prepare test dataset based on case type
    print(f"\nPreparing test dataset (case_type: {config['case_type']})...")
    
    chosen_features_list = []
    rejected_features_list = []
    labels_list = []
    
    if config['case_type'] == 'initial_classification':
        # Mode 1: Classify based on initial reward correctness
        reward_threshold = config.get('reward_threshold', 0.0)
        correct_cases = [c for c in all_cases if c['initial_rewards']['diff'] > reward_threshold]
        incorrect_cases = [c for c in all_cases if c['initial_rewards']['diff'] <= reward_threshold]
        
        print(f"Found {len(correct_cases)} initially correct cases")
        print(f"Found {len(incorrect_cases)} initially incorrect cases")
        
        # Process initially correct cases (label=0)
        for case in tqdm(correct_cases, desc=f"Processing correct cases"):
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, layer, config['max_length'], use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(0)
        
        # Process initially incorrect cases (label=1)
        for case in tqdm(incorrect_cases, desc=f"Processing incorrect cases"):
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, layer, config['max_length'], use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(1)
            
    elif config['case_type'] == 'successfully_attacked':
        # Mode 2: Compare original vs adversarial pairs from successful attacks
        reward_threshold = config.get('reward_threshold', 0.0)
        attacked_cases = filter_successfully_attacked_cases(all_cases, reward_threshold)
        print(f"Found {len(attacked_cases)} successfully attacked cases")
        
        for case in tqdm(attacked_cases, desc=f"Extracting features"):
            # Original pair (label=0)
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, layer, config['max_length'], use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(0)
            
            # Adversarial pair (label=1)
            if 'hacked_chosen' in case and 'hacked_rejected' in case:
                chosen_feat, rejected_feat = extract_pair_features(
                    case['hacked_chosen'], 
                    case['hacked_rejected'],
                    rm_model, tokenizer, sae, layer, config['max_length'], use_sae, use_chat_template
                )
            else:
                chosen_feat, rejected_feat = extract_pair_features(
                    case['adv_chosen'], 
                    case['adv_rejected'],
                    rm_model, tokenizer, sae, layer, config['max_length'], use_sae, use_chat_template
                )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(1)
    
    # Convert to numpy arrays
    X_chosen = np.array(chosen_features_list)
    X_rejected = np.array(rejected_features_list)
    y = np.array(labels_list)

    # Apply scalers if present (legacy checkpoints trained with StandardScaler)
    if scalers is not None:
        X_chosen   = scalers["scaler_chosen"].transform(X_chosen)
        X_rejected = scalers["scaler_rejected"].transform(X_rejected)

    print(f"\nTotal test samples: {len(y)}")
    print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

    # Create DataLoader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_chosen),
        torch.FloatTensor(X_rejected),
        torch.FloatTensor(y)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate
    print("\n--- Evaluating Pre-trained Classifier ---")
    classifier.eval()
    test_preds = []
    test_probs = []
    test_true = []
    
    with torch.no_grad():
        for batch_chosen, batch_rejected, batch_y in test_loader:
            batch_chosen = batch_chosen.to(device)
            batch_rejected = batch_rejected.to(device)
            batch_y = batch_y.to(device)
            outputs = classifier(batch_chosen, batch_rejected).squeeze()
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).view(-1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds)
    test_recall = recall_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    test_auc = roc_auc_score(test_true, test_probs)
    conf_matrix = confusion_matrix(test_true, test_preds)
    
    print("\n" + "="*70)
    print(f"     PRE-TRAINED CLASSIFIER RESULTS ({feature_type.upper()} FEATURES)")
    print("="*70)
    print(f"Model: {os.path.basename(args.pretrained_path)}")
    print(f"Dataset: {os.path.basename(args.dataset_path)}")
    print(f"Case type: {config['case_type']}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {test_precision:.4f}")
    print(f"Recall:         {test_recall:.4f}")
    print(f"F1-Score:       {test_f1:.4f}")
    print(f"AUC-ROC:        {test_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted 0  Predicted 1")
    print(f"Actual 0            {conf_matrix[0,0]:4d}         {conf_matrix[0,1]:4d}")
    print(f"Actual 1            {conf_matrix[1,0]:4d}         {conf_matrix[1,1]:4d}")
    
    # Prepare results
    results = {
        "evaluation_info": {
            "pretrained_model_path": args.pretrained_path,
            "evaluation_dataset": args.dataset_path,
            "feature_type": feature_type.lower(),
            "layer": layer,
            "case_type": config['case_type'],
            "use_chat_template": use_chat_template
        },
        "original_training_config": config,
        "performance_metrics": {
            "test_accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "auc_roc": float(test_auc),
            "confusion_matrix": conf_matrix.tolist(),
            "test_samples": len(y)
        },
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    return results

# --- Training Function ---
def train_classifier(args, all_cases, rm_model, tokenizer, sae=None, use_sae=True, use_chat_template=False):
    """Train classifier with either SAE or raw features and save the model"""
    
    feature_type = "SAE" if use_sae else "Raw"

    if use_sae:
        feature_dim = sae.cfg.d_sae
    else:
        feature_dim = rm_model.config.hidden_size
    
    chosen_features_list = []
    rejected_features_list = []
    labels_list = []
    
    if args.case_type == 'initial_classification':
        correct_cases = [c for c in all_cases if c['initial_rewards']['diff'] > args.reward_threshold]
        incorrect_cases = [c for c in all_cases if c['initial_rewards']['diff'] <= args.reward_threshold]

        if args.balance_classes:
            min_samples = min(len(correct_cases), len(incorrect_cases))
            correct_to_use = correct_cases[:min_samples]
            incorrect_to_use = incorrect_cases[:min_samples]
        else:
            correct_to_use = correct_cases
            incorrect_to_use = incorrect_cases
        
        # Process initially correct cases (label=0)
        for case in tqdm(correct_to_use, desc=f"Processing correct cases ({feature_type})"):
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, args.layer, args.max_length, use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(0)
        
        # Process initially incorrect cases (label=1)
        for case in tqdm(incorrect_to_use, desc=f"Processing incorrect cases ({feature_type})"):
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, args.layer, args.max_length, use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(1)
            
    elif args.case_type == 'successfully_attacked':
        attacked_cases = filter_successfully_attacked_cases(all_cases, args.reward_threshold)

        if len(attacked_cases) == 0:
            print("ERROR: No successfully attacked cases found!")
            return None
        
        for case in tqdm(attacked_cases, desc=f"Extracting features ({feature_type})"):
            # Original pair (label=0)
            chosen_feat, rejected_feat = extract_pair_features(
                case['original_chosen'], 
                case['original_rejected'],
                rm_model, tokenizer, sae, args.layer, args.max_length, use_sae, use_chat_template
            )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(0)
            
            # Adversarial pair (label=1)
            if 'hacked_chosen' in case and 'hacked_rejected' in case:
                chosen_feat, rejected_feat = extract_pair_features(
                    case['hacked_chosen'], 
                    case['hacked_rejected'],
                    rm_model, tokenizer, sae, args.layer, args.max_length, use_sae, use_chat_template
                )
            else:
                chosen_feat, rejected_feat = extract_pair_features(
                    case['adv_chosen'], 
                    case['adv_rejected'],
                    rm_model, tokenizer, sae, args.layer, args.max_length, use_sae, use_chat_template
                )
            chosen_features_list.append(chosen_feat)
            rejected_features_list.append(rejected_feat)
            labels_list.append(1)
    
    # Convert to numpy arrays
    X_chosen = np.array(chosen_features_list)
    X_rejected = np.array(rejected_features_list)
    y = np.array(labels_list)

    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=args.test_ratio, random_state=args.seed, stratify=y
    )
    
    # Get features for each split
    X_chosen_train, X_rejected_train = X_chosen[idx_train], X_rejected[idx_train]
    X_chosen_test, X_rejected_test = X_chosen[idx_test], X_rejected[idx_test]

    # --- Create DataLoaders ---
    train_dataset = TensorDataset(
        torch.FloatTensor(X_chosen_train),
        torch.FloatTensor(X_rejected_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_chosen_test),
        torch.FloatTensor(X_rejected_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    classifier = SAEClassifier(
        sae_dim=feature_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # --- Training Loop ---
    train_losses = []
    best_train_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.num_epochs):
        # Training
        classifier.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        for batch_chosen, batch_rejected, batch_y in train_loader:
            batch_chosen = batch_chosen.to(device)
            batch_rejected = batch_rejected.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_chosen, batch_rejected).squeeze(-1)
            ce_loss = criterion(outputs, batch_y)
            loss = ce_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = accuracy_score(train_true, train_preds)
        
        scheduler.step(avg_train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}")
        
        # Save best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            best_model_state = classifier.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # --- Load Best Model ---
    classifier.load_state_dict(best_model_state)
    
    classifier.eval()
    test_preds = []
    test_probs = []
    test_true = []
    
    with torch.no_grad():
        for batch_chosen, batch_rejected, batch_y in test_loader:
            batch_chosen = batch_chosen.to(device)
            batch_rejected = batch_rejected.to(device)
            batch_y = batch_y.to(device)
            
            outputs = classifier(batch_chosen, batch_rejected).squeeze()
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).view(-1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())
    
    # --- Evaluation Metrics ---
    test_acc = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds)
    test_recall = recall_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    test_auc = roc_auc_score(test_true, test_probs)
    conf_matrix = confusion_matrix(test_true, test_preds)
    
    print("\n" + "="*70)
    print(f"     CLASSIFICATION RESULTS ({feature_type.upper()} FEATURES)     ")
    print("="*70)
    print(f"Mode: {args.case_type}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {test_precision:.4f}")
    print(f"Recall:         {test_recall:.4f}")
    print(f"F1-Score:       {test_f1:.4f}")
    print(f"AUC-ROC:        {test_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted 0  Predicted 1")
    print(f"Actual 0            {conf_matrix[0,0]:4d}         {conf_matrix[0,1]:4d}")
    print(f"Actual 1            {conf_matrix[1,0]:4d}         {conf_matrix[1,1]:4d}")
    
    # --- Save Trained Model and Components ---
    if args.save_model:
        # Generate model identifier
        model_short = args.model_name.split('/')[-1]
        dataset_short = os.path.basename(args.dataset_path).replace('.json', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save directory
        save_dir = os.path.join(args.output_dir, f"{model_short}_{dataset_short}_{feature_type.lower()}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save classifier model
        model_path = os.path.join(save_dir, "classifier.pth")
        torch.save(classifier.state_dict(), model_path)
        print(f"\n✅ Saved classifier to {model_path}")
        
        # Save configuration
        config = {
            'model_name': args.model_name,
            'dataset_path': args.dataset_path,
            'feature_type': feature_type.lower(),
            'feature_dim': int(feature_dim),
            'use_sae': use_sae,
            'use_chat_template': use_chat_template,
            'sae_path': args.sae_path if use_sae else None,
            'layer': args.layer,
            'case_type': args.case_type,
            'reward_threshold': args.reward_threshold,
            'balance_classes': args.balance_classes,
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout,
            'max_length': args.max_length,
            'total_samples': len(y),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'train_ratio': args.train_ratio,
            'test_ratio': args.test_ratio,
            'timestamp': timestamp
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved configuration to {config_path}")
    
    # Prepare results dictionary
    results = {
        "experiment_config": {
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "case_type": args.case_type,
            "feature_type": feature_type.lower(),
            "reward_threshold": args.reward_threshold,
            "use_chat_template": use_chat_template
        },
        "model_config": {
            "layer": args.layer,
            "feature_dim": int(feature_dim),
            "architecture": "cosine_similarity + abs_diff",
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "max_length": args.max_length
        },
        "training_config": {
            "total_sample_pairs": len(y),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_ratio": args.train_ratio,
            "test_ratio": args.test_ratio,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": epoch + 1,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed
        },
        "performance_metrics": {
            "test_accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "auc_roc": float(test_auc),
            "confusion_matrix": conf_matrix.tolist(),
            "best_train_loss": float(best_train_loss)
        },
        "saved_model_path": save_dir if args.save_model else None
    }
    
    return results

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='SAE-based instability detection')

    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the reward model')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for models')
    parser.add_argument('--sae_path', type=str,
                        help='Path to SAE checkpoint (required for SAE features)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_paths', type=str, nargs='+', default=None,
                        help='One or more dataset JSON files; cases are pooled before split')
    parser.add_argument('--output_dir', type=str, default="./classifier_models",
                        help='Directory to save trained classifiers')
    
    parser.add_argument('--dataset_tag', type=str, default=None,
                        help='Tag for output filenames when using --dataset_paths')
    parser.add_argument('--pretrained_path', type=str,
                        help='Path to pre-trained classifier directory (if provided, will evaluate instead of train)')
    
    # Feature type selection
    parser.add_argument('--feature_type', type=str, default='both',
                        choices=['sae', 'raw', 'both'],
                        help='Type of features to use')
    
    # SAE layer configuration
    parser.add_argument('--layer', type=int, default=12,
                        help='Layer index for feature extraction')
    
    # Case selection
    parser.add_argument('--case_type', type=str, default='successfully_attacked',
                        choices=['initial_classification', 'successfully_attacked'],
                        help='Classification mode')
    parser.add_argument('--reward_threshold', type=float, default=0.0,
                        help='Threshold for reward difference')
    
    # Training parameters
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Ratio of data for testing')
    parser.add_argument('--balance_classes', action='store_true', default=True,
                        help='Balance classes by using equal samples from each class')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Model architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128],
                        help='Hidden dimensions for classifier')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Text processing
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save the trained model')
    
    return parser.parse_args()

# --- Main Function ---
def _load_cases(args):
    if args.dataset_paths:
        all_cases = []
        for path in args.dataset_paths:
            with open(path, 'r') as f:
                cases = json.load(f)
            all_cases.extend(cases)
            print(f"  {len(cases)} cases from {os.path.basename(path)}")
        if args.dataset_tag is None:
            args.dataset_path = f"multi{len(args.dataset_paths)}.json"
        else:
            args.dataset_path = args.dataset_tag + ".json"
        return all_cases
    with open(args.dataset_path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()

    if not args.dataset_path and not args.dataset_paths:
        raise ValueError("Provide --dataset_path or --dataset_paths")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    use_chat_template = detect_chat_template_model(args.model_name)
    
    # Check if using pre-trained classifier
    if args.pretrained_path:
        print("\n🔍 Pre-trained classifier detected!")
        print(f"Path: {args.pretrained_path}")
        
        # Check if the path exists
        if not os.path.exists(args.pretrained_path):
            print(f"ERROR: Pre-trained classifier path does not exist: {args.pretrained_path}")
            return
        
        # Check for required files
        required_files = ['classifier.pth', 'config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(args.pretrained_path, file)):
                print(f"ERROR: Required file missing in pre-trained model directory: {file}")
                return
        
        # Load configuration to get model requirements
        with open(os.path.join(args.pretrained_path, 'config.json'), 'r') as f:
            pretrained_config = json.load(f)
        
        # Use model from pretrained config if not specified
        if not args.model_name or args.model_name == "ethz-spylab/poisoned-reward-7b-SUDO-10":
            args.model_name = pretrained_config['model_name']
            print(f"Using model from pre-trained config: {args.model_name}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = load_tokenizer(args.model_name, args.cache_dir)
        
        # Load model
        print("Loading reward model...")
        rm_model, _ = load_model(args.model_name, args.cache_dir)
        
        # Load dataset
        print("Loading dataset...")
        with open(args.dataset_path, 'r') as f:
            all_cases = json.load(f)
        print(f"Loaded {len(all_cases)} total cases from dataset.")
        
        # Load SAE if needed
        sae = None
        if pretrained_config['use_sae']:
            sae_path = pretrained_config.get('sae_path', args.sae_path)
            if not sae_path:
                print("ERROR: SAE is required for this pre-trained model but no path provided!")
                return
            print(f"\nLoading SAE from {sae_path}")
            sae = SAE.load_from_pretrained(sae_path, device="cpu").to(device)
        
        # Evaluate pre-trained classifier
        eval_results = evaluate_pretrained_classifier(args, all_cases, rm_model, tokenizer, sae, use_chat_template)
        
        # Save evaluation results
        model_short = args.model_name.split('/')[-1]
        dataset_short = os.path.basename(args.dataset_path).replace('.json', '')
        output_file = os.path.join(args.output_dir, 
                                   f"pretrained_eval_{model_short}_{dataset_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n✅ Saved evaluation results to {output_file}")
        
        print("\n✅ Evaluation completed successfully!")
        return
    else:
        # Otherwise, proceed with training
        print(f"Classification mode: {args.case_type}")
        print(f"Feature type(s): {args.feature_type}")
        print()
        
        model_type = detect_model_type(args.model_name)
        tokenizer = load_tokenizer(args.model_name, args.cache_dir)
        rm_model, _ = load_model(args.model_name, args.cache_dir, model_type)

        all_cases = _load_cases(args)
        print(f"Loaded {len(all_cases)} total cases.")

        all_results = {}

        if args.feature_type in ['sae', 'both']:
            if not args.sae_path:
                print("ERROR: SAE path is required for SAE features!")
                if args.feature_type == 'sae':
                    return
            else:
                sae = SAE.load_from_pretrained(args.sae_path, device="cpu").to(device)
                
                # Train with SAE features
                sae_results = train_classifier(args, all_cases, rm_model, tokenizer, sae, 
                                              use_sae=True, use_chat_template=use_chat_template)
                if sae_results:
                    all_results['sae'] = sae_results
                    
                    # Save SAE results
                    model_short = args.model_name.split('/')[-1]
                    dataset_short = os.path.basename(args.dataset_path).replace('.json', '')
                    output_file = os.path.join(args.output_dir, 
                                            f"sae_results_{model_short}_{dataset_short}_layer{args.layer}_{args.case_type}.json")
                    with open(output_file, 'w') as f:
                        json.dump(sae_results, f, indent=2)
                    print(f"✅ Saved SAE results to {output_file}")
        
        if args.feature_type in ['raw', 'both']:
            # Train with raw features
            raw_results = train_classifier(args, all_cases, rm_model, tokenizer, None, 
                                          use_sae=False, use_chat_template=use_chat_template)
            if raw_results:
                all_results['raw'] = raw_results
                
                # Save raw results
                model_short = args.model_name.split('/')[-1]
                dataset_short = os.path.basename(args.dataset_path).replace('.json', '')
                output_file = os.path.join(args.output_dir, 
                                        f"raw_results_{model_short}_{dataset_short}_layer{args.layer}_{args.case_type}.json")
                with open(output_file, 'w') as f:
                    json.dump(raw_results, f, indent=2)
                print(f"✅ Saved raw results to {output_file}")
        
        # Compare results if both were run
        if args.feature_type == 'both' and 'sae' in all_results and 'raw' in all_results:
            print("\n" + "="*70)
            print("     COMPARISON: SAE vs RAW FEATURES")
            print("="*70)
            
            sae_metrics = all_results['sae']['performance_metrics']
            raw_metrics = all_results['raw']['performance_metrics']
            
            print(f"{'Metric':<15} {'SAE':>10} {'Raw':>10} {'Difference':>12}")
            print("-" * 47)
            
            metrics_to_compare = ['test_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            for metric in metrics_to_compare:
                sae_val = sae_metrics[metric]
                raw_val = raw_metrics[metric]
                diff = sae_val - raw_val
                sign = "+" if diff > 0 else ""
                print(f"{metric:<15} {sae_val:>10.4f} {raw_val:>10.4f} {sign}{diff:>11.4f}")
            
            # Save comparison
            comparison = {
                'experiment': args.case_type,
                'model': args.model_name,
                'dataset': args.dataset_path,
                'layer': args.layer,
                'use_chat_template': use_chat_template,
                'sae_results': all_results['sae'],
                'raw_results': all_results['raw'],
                'comparison': {
                    'accuracy_improvement': sae_metrics['test_accuracy'] - raw_metrics['test_accuracy'],
                    'f1_improvement': sae_metrics['f1_score'] - raw_metrics['f1_score'],
                    'auc_improvement': sae_metrics['auc_roc'] - raw_metrics['auc_roc']
                }
            }
            
            model_short = args.model_name.split('/')[-1]
            dataset_short = os.path.basename(args.dataset_path).replace('.json', '')
            comparison_file = os.path.join(args.output_dir, 
                                        f"comparison_{model_short}_{dataset_short}_layer{args.layer}_{args.case_type}.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n✅ Saved comparison to {comparison_file}")
    
    print("\n✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()