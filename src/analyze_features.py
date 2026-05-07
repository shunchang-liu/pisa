import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, LlamaModel, LlamaPreTrainedModel, LlamaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from sae_lens import SAE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

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
    """Detect the type of model based on its name."""
    model_name_lower = model_name.lower()
    
    chat_template_models = ['skywork', 'reward-v2', 'chat', 'instruct']
    if any(template_key in model_name_lower for template_key in chat_template_models):
        return 'chat_template'
    
    if 'poisoned' in model_name_lower or 'beaver' in model_name_lower or 'rm' in model_name_lower:
        return 'score_model'
    
    return 'standard'


def load_tokenizer(model_name: str, cache_dir: str):
    """Load the appropriate tokenizer for the model."""
    model_name_lower = model_name.lower()
    
    llama_based_models = [
        'poisoned-rlhf', 'poisoned-reward', 'alpaca', 'vicuna', 'koala',
        'wizardlm', 'guanaco'
    ]
    
    use_llama_tokenizer = False
    
    if 'llama' in model_name_lower:
        use_llama_tokenizer = True
    else:
        for llama_variant in llama_based_models:
            if llama_variant in model_name_lower:
                use_llama_tokenizer = True
                break
        
        if not use_llama_tokenizer:
            try:
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
                if hasattr(config, 'model_type') and 'llama' in config.model_type.lower():
                    use_llama_tokenizer = True
            except:
                pass
    
    if use_llama_tokenizer:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"Using LlamaTokenizer for {model_name}")
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"Fallback to AutoTokenizer for {model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Using AutoTokenizer for {model_name}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_model(model_name: str, cache_dir: str, model_type: str = None):
    """Load the appropriate model based on its type."""
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    print(f"Loading {model_name} as {model_type}")
    
    if model_type == 'score_model':
        model = LlamaForScore.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto" if device.type == "cuda" else None
        )
    elif model_type == 'chat_template':
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
            print(f"Loaded {model_name} with flash_attention_2")
        except:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device.type == "cuda" else None,
                num_labels=1,
                cache_dir=cache_dir
            )
            print(f"Loaded {model_name} with standard attention")
    else:
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


def process_dialogue_to_prompt(dialogue_text: str) -> Tuple[str, str]:
    """Process dialogue text to extract prompt and answer."""
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


def tokenize_pair(chosen: str, rejected: str, tokenizer, max_length: int = 512):
    """Tokenize a pair of texts."""
    
    if "Human:" in chosen and "Assistant:" in chosen:
        chosen_prompt, chosen_answer = process_dialogue_to_prompt(chosen)
        rejected_prompt, rejected_answer = process_dialogue_to_prompt(rejected)
        
        chosen_full = chosen_prompt + chosen_answer
        rejected_full = rejected_prompt + rejected_answer
    else:
        chosen_full = chosen
        rejected_full = rejected
    
    chosen_ids = tokenizer(
        chosen_full,
        add_special_tokens=True,
        truncation=False,
        return_tensors='pt'
    )['input_ids'][0]
    
    rejected_ids = tokenizer(
        rejected_full,
        add_special_tokens=True,
        truncation=False,
        return_tensors='pt'
    )['input_ids'][0]
    
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
        chosen_ids = torch.cat([
            chosen_ids,
            torch.full((pad_len,), pad_token_id, dtype=chosen_ids.dtype)
        ])
        chosen_mask = torch.cat([
            chosen_mask,
            torch.zeros(pad_len, dtype=torch.bool)
        ])
    
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


# --- Case Filtering Functions ---
def filter_successfully_attacked_cases(cases: List[Dict], threshold: float = 0.0) -> List[Dict]:
    """
    Filter cases where attack succeeded (flipped from positive to negative)
    
    """
    attacked = []
    for c in cases:
        try:
            initial_diff = c.get('initial_rewards', {}).get('diff', None)
            final_diff = c.get('final_rewards', {}).get('diff', None)
            
            if initial_diff is not None and final_diff is not None:
                if initial_diff > threshold and final_diff <= -threshold:
                    attacked.append(c)
        except:
            continue
    return attacked


# --- SAE Feature Distribution Analyzer ---
class SAEFeatureDistributionAnalyzer:
    """
    """
    
    def __init__(self, rm_model, tokenizer, sae, layer: int, model_type: str,
                 max_length: int = 512):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer = layer
        self.model_type = model_type
        self.max_length = max_length
        self.device = device
        self.feature_dim = sae.cfg.d_sae
        
        self.benign_chosen_features = []
        self.benign_rejected_features = []
        self.hacking_chosen_features = []
        self.hacking_rejected_features = []

        self.benign_chosen_raw = []
        self.benign_rejected_raw = []
        self.hacking_chosen_raw = []
        self.hacking_rejected_raw = []

        self.feature_stats = None
    
    def extract_sae_features_separate(self, chosen_text: str, rejected_text: str):
        """Extract SAE and raw hidden features for chosen and rejected separately.

        Returns:
            (pooled_sae_ch, pooled_sae_rj, pooled_raw_ch, pooled_raw_rj)
            sae:  [d_sae],  raw: [d_model]
        """
        activation_layer = self.layer + 1

        tokens = tokenize_pair(chosen_text, rejected_text, self.tokenizer, self.max_length)
        chosen_ids  = tokens['chosen_input_ids'].to(self.device)
        chosen_mask = tokens['chosen_attention_mask'].to(self.device)
        rejected_ids  = tokens['rejected_input_ids'].to(self.device)
        rejected_mask = tokens['rejected_attention_mask'].to(self.device)

        with torch.no_grad():
            # Process chosen
            outputs_ch = self.rm_model.model(chosen_ids, attention_mask=chosen_mask,
                                             output_hidden_states=True)
            hidden_ch = outputs_ch.hidden_states[activation_layer]   # [1, seq, d]
            ch_mask = chosen_mask[0].bool()
            raw_seq_ch = hidden_ch[0, ch_mask, :]                    # [seq, d]
            pooled_raw_ch = raw_seq_ch.mean(dim=0)                   # [d]
            sae_features_ch = self.sae.encode(raw_seq_ch)            # [seq, k]
            pooled_sae_ch = sae_features_ch.mean(dim=0)              # [k]

            # Process rejected
            outputs_rj = self.rm_model.model(rejected_ids, attention_mask=rejected_mask,
                                             output_hidden_states=True)
            hidden_rj = outputs_rj.hidden_states[activation_layer]
            rj_mask = rejected_mask[0].bool()
            raw_seq_rj = hidden_rj[0, rj_mask, :]
            pooled_raw_rj = raw_seq_rj.mean(dim=0)
            sae_features_rj = self.sae.encode(raw_seq_rj)
            pooled_sae_rj = sae_features_rj.mean(dim=0)

        return pooled_sae_ch, pooled_sae_rj, pooled_raw_ch, pooled_raw_rj
    
    def collect_features(self, cases: List[Dict]):
        """
        """
        print("\nCollecting features...")
        
        for case in tqdm(cases, desc="Processing samples"):
            hacking_chosen = case.get('adv_chosen', case.get('hacked_chosen'))
            hacking_rejected = case.get('adv_rejected', case.get('hacked_rejected'))

            if hacking_chosen and hacking_rejected:
                sae_ch, sae_rj, raw_ch, raw_rj = self.extract_sae_features_separate(
                    hacking_chosen, hacking_rejected
                )
                self.hacking_chosen_features.append(sae_ch.cpu().numpy())
                self.hacking_rejected_features.append(sae_rj.cpu().numpy())
                self.hacking_chosen_raw.append(raw_ch.cpu().float().numpy())
                self.hacking_rejected_raw.append(raw_rj.cpu().float().numpy())

            if 'original_chosen' in case and 'original_rejected' in case:
                sae_ch, sae_rj, raw_ch, raw_rj = self.extract_sae_features_separate(
                    case['original_chosen'], case['original_rejected']
                )
                self.benign_chosen_features.append(sae_ch.cpu().numpy())
                self.benign_rejected_features.append(sae_rj.cpu().numpy())
                self.benign_chosen_raw.append(raw_ch.cpu().float().numpy())
                self.benign_rejected_raw.append(raw_rj.cpu().float().numpy())
        
        print(f"Collected {len(self.hacking_chosen_features)} hacking samples")
        print(f"Collected {len(self.benign_chosen_features)} benign samples")
    
    def compute_feature_statistics(self, activation_threshold: float = 0.0) -> pd.DataFrame:
        """Compute per-feature statistics for chosen and rejected separately."""
        print("\n" + "="*50)
        print("    Computing Feature Statistics")
        print("="*50)
        
        benign_chosen_array = np.array(self.benign_chosen_features)
        benign_rejected_array = np.array(self.benign_rejected_features)
        hacking_chosen_array = np.array(self.hacking_chosen_features)
        hacking_rejected_array = np.array(self.hacking_rejected_features)
        
        stats_list = []
        
        for i in tqdm(range(self.feature_dim), desc="Computing statistics"):
            # Benign
            benign_chosen_vals = benign_chosen_array[:, i]
            benign_rejected_vals = benign_rejected_array[:, i]
            # Hacking
            hacking_chosen_vals = hacking_chosen_array[:, i]
            hacking_rejected_vals = hacking_rejected_array[:, i]
            
            benign_chosen_act_rate = np.mean(benign_chosen_vals > activation_threshold)
            benign_rejected_act_rate = np.mean(benign_rejected_vals > activation_threshold)
            hacking_chosen_act_rate = np.mean(hacking_chosen_vals > activation_threshold)
            hacking_rejected_act_rate = np.mean(hacking_rejected_vals > activation_threshold)
            
            benign_chosen_mean = np.mean(benign_chosen_vals)
            benign_rejected_mean = np.mean(benign_rejected_vals)
            hacking_chosen_mean = np.mean(hacking_chosen_vals)
            hacking_rejected_mean = np.mean(hacking_rejected_vals)
            
            benign_chosen_std = np.std(benign_chosen_vals)
            benign_rejected_std = np.std(benign_rejected_vals)
            hacking_chosen_std = np.std(hacking_chosen_vals)
            hacking_rejected_std = np.std(hacking_rejected_vals)
            
            stats_list.append({
                'feature_idx': i,
                'benign_chosen_activation_rate': benign_chosen_act_rate,
                'benign_rejected_activation_rate': benign_rejected_act_rate,
                'benign_chosen_mean': benign_chosen_mean,
                'benign_rejected_mean': benign_rejected_mean,
                'benign_chosen_std': benign_chosen_std,
                'benign_rejected_std': benign_rejected_std,
                'hacking_chosen_activation_rate': hacking_chosen_act_rate,
                'hacking_rejected_activation_rate': hacking_rejected_act_rate,
                'hacking_chosen_mean': hacking_chosen_mean,
                'hacking_rejected_mean': hacking_rejected_mean,
                'hacking_chosen_std': hacking_chosen_std,
                'hacking_rejected_std': hacking_rejected_std,
                'benign_chosen_rejected_diff': benign_chosen_act_rate - benign_rejected_act_rate,
                'hacking_chosen_rejected_diff': hacking_chosen_act_rate - hacking_rejected_act_rate,
            })
        
        self.feature_stats = pd.DataFrame(stats_list)
        
        print(f"\nFeature Statistics Summary:")
        print(f"  Total features: {self.feature_dim}")
        print(f"  Benign samples: {len(self.benign_chosen_features)}")
        print(f"  Hacking samples: {len(self.hacking_chosen_features)}")
        
        return self.feature_stats
    
    def visualize_distribution(self, save_dir: str):
        """
        """
        if self.feature_stats is None:
            raise ValueError("Please run compute_feature_statistics() first")

        os.makedirs(save_dir, exist_ok=True)

        COLOR_BENIGN  = '#5BAFF0'   # sky blue  (line)
        COLOR_FRAGILE = '#FFB3D1'   # light pink (vlines)

        n_features = len(self.feature_stats)
        x = np.arange(n_features)

        def _make_chart(ax, benign_vals, fragile_vals, xlabel):
            # Perturbed: vertical lines
            ax.vlines(x, 0, fragile_vals,
                      label='Perturbed', colors=COLOR_FRAGILE,
                      linewidth=2.0, alpha=0.75, zorder=2)
            # Benign: line on top
            ax.plot(x, benign_vals,
                    label='Benign', color=COLOR_BENIGN, linewidth=5.0, alpha=0.9, zorder=3)

            ax.set_xlabel(xlabel, fontsize=28)
            ax.set_ylabel('Activation Rate', fontsize=28)
            ax.legend(loc='upper right', fontsize=28)
            ax.tick_params(axis='both', labelsize=24)
            ax.set_xlim(0, n_features)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)

        # ── Plot 1: Chosen ──────────────────────────────────────────────────
        sorted_chosen = self.feature_stats.sort_values(
            'benign_chosen_activation_rate', ascending=False
        )
        fig, ax = plt.subplots(figsize=(16, 6))
        _make_chart(
            ax,
            sorted_chosen['benign_chosen_activation_rate'].values,
            sorted_chosen['hacking_chosen_activation_rate'].values,
            'Feature Index (sorted by Benign Winning activation rate)'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'activation_rate_chosen.png'), dpi=150)
        plt.close()

        # ── Plot 2: Rejected ────────────────────────────────────────────────
        sorted_rejected = self.feature_stats.sort_values(
            'benign_rejected_activation_rate', ascending=False
        )
        fig, ax = plt.subplots(figsize=(16, 6))
        _make_chart(
            ax,
            sorted_rejected['benign_rejected_activation_rate'].values,
            sorted_rejected['hacking_rejected_activation_rate'].values,
            'Feature Index (sorted by Benign Losing activation rate)'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'activation_rate_rejected.png'), dpi=150)
        plt.close()

        print(f"\nVisualizations saved to {save_dir}/")
        print(f"  - activation_rate_chosen.png")
        print(f"  - activation_rate_rejected.png")
    
    def save_results(self, save_dir: str, extra_summary: Optional[Dict] = None):
        """
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.feature_stats is not None:
            self.feature_stats.to_csv(
                os.path.join(save_dir, 'feature_statistics.csv'),
                index=False
            )

        np.save(os.path.join(save_dir, 'benign_chosen_features.npy'),
                np.array(self.benign_chosen_features))
        np.save(os.path.join(save_dir, 'benign_rejected_features.npy'),
                np.array(self.benign_rejected_features))
        np.save(os.path.join(save_dir, 'hacking_chosen_features.npy'),
                np.array(self.hacking_chosen_features))
        np.save(os.path.join(save_dir, 'hacking_rejected_features.npy'),
                np.array(self.hacking_rejected_features))

        np.save(os.path.join(save_dir, 'benign_chosen_raw.npy'),
                np.array(self.benign_chosen_raw))
        np.save(os.path.join(save_dir, 'benign_rejected_raw.npy'),
                np.array(self.benign_rejected_raw))
        np.save(os.path.join(save_dir, 'hacking_chosen_raw.npy'),
                np.array(self.hacking_chosen_raw))
        np.save(os.path.join(save_dir, 'hacking_rejected_raw.npy'),
                np.array(self.hacking_rejected_raw))

        summary = {
            'n_features': self.feature_dim,
            'n_benign_samples': len(self.benign_chosen_features),
            'n_hacking_samples': len(self.hacking_chosen_features),
            'layer': self.layer,
            'max_length': self.max_length,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
        }
        if extra_summary:
            summary.update(extra_summary)

        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {save_dir}")


# --- Main Function ---
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SAE Feature Distribution Analysis')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default=None
                       help='Cache directory for models')
    parser.add_argument('--sae_path', type=str, required=True, help='Path to SAE model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset JSON')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results. If omitted, derived as '
                            '<results_root>/<run_name>')
    parser.add_argument('--results_root', type=str, default='./sae_distribution_analysis',
                       help='Root directory when --output_dir is not given')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Subdirectory name under --results_root. '
                            'Defaults to <model_tag>__<dataset_tag>')
    parser.add_argument('--model_tag', type=str, default=None,
                       help='Short tag for the model (used to build run_name)')
    parser.add_argument('--dataset_tag', type=str, default=None,
                       help='Short tag for the dataset (used to build run_name)')
    parser.add_argument('--layer', type=int, default=12, help='Layer to extract features from')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--activation_threshold', type=float, default=0.0,
                       help='Threshold for determining feature activation')
    parser.add_argument('--reward_threshold', type=float, default=0.0,
                       help='Threshold for filtering successfully attacked cases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        if args.run_name is None:
            model_tag = args.model_tag or os.path.basename(args.model_name.rstrip('/'))
            dataset_tag = args.dataset_tag or os.path.splitext(
                os.path.basename(args.dataset_path))[0]
            args.run_name = f"{model_tag}__{dataset_tag}"
        args.output_dir = os.path.join(args.results_root, args.run_name)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*70)
    print("    SAE FEATURE DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nModel: {args.model_name}")
    print(f"SAE: {args.sae_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Layer: {args.layer}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_type = detect_model_type(args.model_name)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    rm_model, _ = load_model(args.model_name, args.cache_dir, model_type)
    
    # Load SAE
    print("Loading SAE...")
    sae = SAE.load_from_pretrained(args.sae_path, device="cpu").to(device)
    
    # Initialize analyzer
    analyzer = SAEFeatureDistributionAnalyzer(
        rm_model=rm_model,
        tokenizer=tokenizer,
        sae=sae,
        layer=args.layer,
        model_type=model_type,
        max_length=args.max_length
    )
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        all_cases = json.load(f)
    
    print(f"Loaded {len(all_cases)} total cases")
    
    # Filter successfully attacked cases
    attacked_cases = filter_successfully_attacked_cases(all_cases, threshold=args.reward_threshold)
    print(f"Found {len(attacked_cases)} successfully attacked cases (threshold={args.reward_threshold})")
    
    if len(attacked_cases) == 0:
        print("ERROR: No successfully attacked cases found!")
        return
    
    # Collect features
    analyzer.collect_features(attacked_cases)
    
    # Compute statistics
    analyzer.compute_feature_statistics(activation_threshold=args.activation_threshold)
    
    # Visualize results
    analyzer.visualize_distribution(args.output_dir)

    # Save results
    analyzer.save_results(args.output_dir, extra_summary={
        'model_name': args.model_name,
        'sae_path': args.sae_path,
        'dataset_path': args.dataset_path,
        'model_tag': args.model_tag,
        'dataset_tag': args.dataset_tag,
        'run_name': args.run_name,
        'activation_threshold': args.activation_threshold,
        'reward_threshold': args.reward_threshold,
    })

    print("\n✅ Analysis completed successfully!")


if __name__ == "__main__":
    main()