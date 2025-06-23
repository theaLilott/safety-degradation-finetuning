import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
import gc
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import evaluate
import numpy as np


class UnifiedPerformanceEvaluator:
    """
    Unified Performance evaluator that handles LoRA, QLoRA, and full fine-tuning checkpoints
    with BLEU score evaluation on English-to-Spanish translation
    """
    def __init__(self, 
                 base_model_name: str, 
                 device: str = "auto", 
                 val_size: int = 1000,
                 val_start_idx: int = 5000,
                 random_seed: int = 42):
        """
        Initialize the unified performance evaluator.
        
        Args:
            base_model_name: The base model name or path for tokenizer and base model loading
            device: Device to run inference on
            val_size: Number of validation examples to use
            val_start_idx: Starting index for validation set (should be after training data)
            random_seed: Random seed for reproducible sampling
        """
        self.base_model_name = base_model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.val_size = val_size
        self.val_start_idx = val_start_idx
        self.random_seed = random_seed
        
        # Load tokenizer
        print(f"Loading tokenizer from: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load evaluation metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
        # Load evaluation dataset
        self.load_dataset()
        
        print(f"Performance evaluator initialized with {len(self.dataset)} validation examples")

    def load_dataset(self):
        """Load the English-to-Spanish validation dataset"""
        try:
            # Load the dataset
            ds = load_dataset("okezieowen/english_to_spanish")
            full_dataset = pd.DataFrame(ds['train'])
            print(f"Loaded full English-to-Spanish dataset with {len(full_dataset)} samples")
            
            # Verify required columns exist
            if 'English' not in full_dataset.columns or 'Spanish' not in full_dataset.columns:
                raise ValueError("Dataset must contain 'English' and 'Spanish' columns")
            
            # Create validation set starting from val_start_idx
            val_end_idx = self.val_start_idx + self.val_size
            if val_end_idx > len(full_dataset):
                print(f"Warning: Requested validation set goes beyond dataset size. Using examples {self.val_start_idx}:{len(full_dataset)}")
                val_end_idx = len(full_dataset)
            
            self.dataset = full_dataset.iloc[self.val_start_idx:val_end_idx].reset_index(drop=True)
            print(f"Using validation examples {self.val_start_idx}:{val_end_idx} ({len(self.dataset)} total)")
            
            # Show some examples
            print("\nExample validation pairs:")
            for i in range(min(3, len(self.dataset))):
                print(f"  EN: {self.dataset.iloc[i]['English']}")
                print(f"  ES: {self.dataset.iloc[i]['Spanish']}")
                print()
            
        except Exception as e:
            print(f"Error loading English-to-Spanish dataset: {e}")
            raise

    def detect_checkpoint_type(self, checkpoint_path: Path) -> str:
        """Detect whether checkpoint is LoRA, QLoRA, or full fine-tuning"""
        if (checkpoint_path / "adapter_config.json").exists():
            # Read adapter config to determine if it's QLoRA
            try:
                with open(checkpoint_path / "adapter_config.json", 'r') as f:
                    config = json.load(f)
                # Check for quantization indicators
                if (config.get("quantization_config") is not None or 
                    "bnb" in str(config).lower() or 
                    "4bit" in str(config).lower() or
                    "8bit" in str(config).lower()):
                    return "qlora"
                return "lora"
            except:
                return "lora"  # Default to LoRA if can't read config
        elif (checkpoint_path / "config.json").exists():
            return "full"
        else:
            raise ValueError(f"Cannot determine checkpoint type for {checkpoint_path}")

    def get_checkpoint_paths(self, checkpoint_dir: str) -> List[Tuple[Path, str]]:
        """Get all checkpoint directories with their types"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = []
        
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                try:
                    checkpoint_type = self.detect_checkpoint_type(item)
                    checkpoints.append((item, checkpoint_type))
                    print(f"Found {checkpoint_type} checkpoint: {item.name}")
                except ValueError as e:
                    print(f"Skipping {item}: {e}")
                    continue
        
        # Sort by checkpoint number if they follow a naming pattern
        try:
            checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x[0].name)[-1]))
        except (IndexError, TypeError):
            print("Could not sort checkpoints by number, sorting alphabetically.")
            checkpoints.sort(key=lambda x: x[0].name)
            
        return checkpoints

    def load_checkpoint(self, checkpoint_path: Path, checkpoint_type: str):
        """Load a checkpoint based on its type - loads fresh base model for each checkpoint"""
        try:
            print(f"Loading {checkpoint_type} checkpoint: {checkpoint_path.name}")
            
            if checkpoint_type == "full":
                # Full fine-tuning checkpoint
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
                return model
                
            elif checkpoint_type == "qlora":
                # QLoRA checkpoint - load fresh quantized base model
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,                      
                    bnb_4bit_use_double_quant=True,         
                    bnb_4bit_quant_type="nf4",              
                    bnb_4bit_compute_dtype=torch.bfloat16,  
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
                
                # Apply adapter to fresh base model
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                return model
                
            elif checkpoint_type == "lora":
                # Regular LoRA checkpoint - load fresh base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
                
                # Apply adapter to fresh base model
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                return model
                
            else:
                raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
                
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None

    def format_translation_prompt(self, spanish_text: str) -> str:
        """Format the English text as a translation prompt"""
        return f"Translate the following Spanish text to English: {spanish_text}"

    def generate_translation(self, model, spanish_text: str, max_length: int = 128, temperature: float = 0.3) -> str:
        """Generate English translation from Spanish text"""
        try:
            # Create translation prompt
            prompt = self.format_translation_prompt(spanish_text)
            
            # Format using chat template - adjust this based on your model's expected format
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize and generate
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_length, 
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            translation = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return translation
            
        except Exception as e:
            print(f"Error generating translation: {e}")
            return ""

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU and ROUGE scores and related metrics"""
        try:
            # Ensure we have valid predictions and references
            valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) 
                          if pred.strip() and ref.strip()]
            
            if not valid_pairs:
                return {
                    "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0,
                    "valid_translations": 0, "total_examples": len(predictions)
                }
            
            valid_predictions, valid_references = zip(*valid_pairs)
            
            # Compute BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=list(valid_predictions), 
                references=[[ref] for ref in valid_references]  # BLEU expects list of lists for references
            )
            
            # Compute ROUGE scores
            rouge_result = self.rouge_metric.compute(
                predictions=list(valid_predictions),
                references=list(valid_references)
            )
            
            return {
                # BLEU metrics
                "bleu": bleu_result["bleu"],
                "bleu_1": bleu_result["precisions"][0] if len(bleu_result["precisions"]) > 0 else 0.0,
                "bleu_2": bleu_result["precisions"][1] if len(bleu_result["precisions"]) > 1 else 0.0,
                "bleu_3": bleu_result["precisions"][2] if len(bleu_result["precisions"]) > 2 else 0.0,
                "bleu_4": bleu_result["precisions"][3] if len(bleu_result["precisions"]) > 3 else 0.0,
                "brevity_penalty": bleu_result["brevity_penalty"],
                "length_ratio": bleu_result["length_ratio"],
                
                # ROUGE metrics
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"],
                "rougeLsum": rouge_result["rougeLsum"],
                
                # Summary stats
                "valid_translations": len(valid_pairs),
                "total_examples": len(predictions)
            }
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {
                "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0,
                "error": str(e), "valid_translations": 0, "total_examples": len(predictions)
            }

    def load_existing_results(self, output_file: str) -> List[Dict]:
        """Load existing results from CSV file"""
        try:
            if os.path.exists(output_file):
                return pd.read_csv(output_file).to_dict('records')
            return []
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
            return []

    def append_results(self, results: List[Dict], output_file: str):
        """Append results to CSV file"""
        try:
            df_new = pd.DataFrame(results)
            if os.path.exists(output_file):
                df_new.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_new.to_csv(output_file, mode='w', header=True, index=False)
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")

    def save_checkpoint_summary(self, checkpoint_summaries: List[Dict], summary_file: str = "performance_checkpoint_summary.json"):
        """Save checkpoint summary to JSON file"""
        try:
            with open(summary_file, 'w') as f:
                json.dump(checkpoint_summaries, f, indent=2)
            print(f"Checkpoint summary saved to {summary_file}")
        except Exception as e:
            print(f"Error saving checkpoint summary: {e}")

    def get_completed_checkpoints(self, output_file: str) -> set:
        """Get set of completed checkpoint names"""
        existing_results = self.load_existing_results(output_file)
        if not existing_results:
            return set()
        
        # Count successful evaluations per checkpoint
        checkpoint_counts = {}
        for result in existing_results:
            checkpoint = result['checkpoint']
            checkpoint_counts.setdefault(checkpoint, 0)
            if 'error' not in str(result.get('translation', '')).lower():
                checkpoint_counts[checkpoint] += 1
        
        # Return checkpoints that have been fully evaluated
        total_examples = len(self.dataset)
        return {cp for cp, count in checkpoint_counts.items() if count >= total_examples}

    def evaluate_checkpoint(self, 
                          model, 
                          checkpoint_name: str, 
                          checkpoint_type: str,
                          batch_size: int = 4, 
                          output_file: str = "performance_results.csv", 
                          save_frequency: int = 50) -> Tuple[List[Dict], Dict[str, float]]:
        """Evaluate a single checkpoint"""
        results = []
        predictions = []
        references = []
        
        print(f"Evaluating {checkpoint_type} checkpoint: {checkpoint_name}")
        
        # Check for existing results
        existing_results = self.load_existing_results(output_file)
        completed_examples = {(r['checkpoint'], r['spanish_text']) for r in existing_results}
        
        already_completed = len([r for r in existing_results if r['checkpoint'] == checkpoint_name])
        print(f"Found {already_completed} already completed examples for {checkpoint_name}")
        
        processed_count = 0
        
        for i in tqdm(range(0, len(self.dataset), batch_size), desc="Processing batches"):
            batch = self.dataset.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                spanish_text = row['Spanish']
                english_reference = row['English']
                
                if (checkpoint_name, spanish_text) in completed_examples:
                    continue
                
                try:
                    # Generate translation
                    translation = self.generate_translation(model, spanish_text)
                    
                    # Create result record
                    result = {
                        'checkpoint': checkpoint_name,
                        'checkpoint_type': checkpoint_type,
                        'spanish_text': spanish_text,
                        'english_reference': english_reference,
                        'translation': translation,
                        'timestamp': datetime.now().isoformat(),
                        'val_size': len(self.dataset),
                        'val_start_idx': self.val_start_idx
                    }
                    
                    results.append(result)
                    predictions.append(translation)
                    references.append(english_reference)
                    
                    processed_count += 1
                    
                    # Save intermediate results
                    if processed_count % save_frequency == 0:
                        self.append_results(results[-save_frequency:], output_file)
                    
                except Exception as e:
                    print(f"Error processing example for {checkpoint_name}: {e}")
                    error_result = {
                        'checkpoint': checkpoint_name,
                        'checkpoint_type': checkpoint_type,
                        'spanish_text': spanish_text,
                        'english_reference': english_reference,
                        'translation': f"ERROR: {str(e)}",
                        'timestamp': datetime.now().isoformat(),
                        'val_size': len(self.dataset),
                        'val_start_idx': self.val_start_idx
                    }
                    self.append_results([error_result], output_file)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save remaining results
        if results and (processed_count % save_frequency != 0):
            self.append_results(results[-(processed_count % save_frequency):], output_file)
        
        # Compute BLEU and ROUGE scores
        metrics = self.compute_metrics(predictions, references)
        
        print(f"Checkpoint {checkpoint_name} ({checkpoint_type}) - BLEU: {metrics['bleu']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  Valid translations: {metrics['valid_translations']}/{metrics['total_examples']}")
        
        return results, metrics

    def run_evaluation(self, 
                      checkpoint_dir: str, 
                      output_file: str = "performance_unified_results.csv", 
                      batch_size: int = 4, 
                      save_frequency: int = 50, 
                      resume: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Run performance evaluation on all checkpoints"""
        checkpoint_paths = self.get_checkpoint_paths(checkpoint_dir)
        
        if not checkpoint_paths:
            print(f"No valid checkpoints found in {checkpoint_dir}")
            return [], []
        
        print(f"Found {len(checkpoint_paths)} checkpoints to evaluate:")
        for path, ctype in checkpoint_paths:
            print(f"  {path.name}: {ctype}")
        
        # Check for completed checkpoints
        completed_checkpoints = self.get_completed_checkpoints(output_file) if resume else set()
        if completed_checkpoints:
            print(f"Resuming evaluation. Found {len(completed_checkpoints)} completed checkpoints.")
        
        all_results = []
        checkpoint_summaries = []
        
        # Load existing summaries
        summary_file = output_file.replace('.csv', '_summary.json')
        if resume and os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    checkpoint_summaries = json.load(f)
            except Exception as e:
                print(f"Could not load existing summaries: {e}")
        
        for checkpoint_path, checkpoint_type in checkpoint_paths:
            checkpoint_name = checkpoint_path.name
            
            if checkpoint_name in completed_checkpoints:
                print(f"Skipping already completed checkpoint: {checkpoint_name}")
                continue
            
            print(f"\n{'='*60}\nStarting performance evaluation of {checkpoint_type} checkpoint: {checkpoint_name}\n{'='*60}")
            
            model = self.load_checkpoint(checkpoint_path, checkpoint_type)
            if model is None:
                continue
            
            try:
                results, metrics = self.evaluate_checkpoint(
                    model, checkpoint_name, checkpoint_type, batch_size, output_file, save_frequency
                )
                
                all_results.extend(results)
                
                # Create summary
                summary = {
                    'checkpoint': checkpoint_name,
                    'checkpoint_type': checkpoint_type,
                    # BLEU metrics
                    'bleu_score': metrics['bleu'],
                    'bleu_1': metrics.get('bleu_1', 0.0),
                    'bleu_2': metrics.get('bleu_2', 0.0),
                    'bleu_3': metrics.get('bleu_3', 0.0),
                    'bleu_4': metrics.get('bleu_4', 0.0),
                    'brevity_penalty': metrics.get('brevity_penalty', 0.0),
                    'length_ratio': metrics.get('length_ratio', 0.0),
                    # ROUGE metrics
                    'rouge1': metrics.get('rouge1', 0.0),
                    'rouge2': metrics.get('rouge2', 0.0),
                    'rougeL': metrics.get('rougeL', 0.0),
                    'rougeLsum': metrics.get('rougeLsum', 0.0),
                    # Summary stats
                    'valid_translations': metrics['valid_translations'],
                    'total_examples': metrics['total_examples'],
                    'completed_at': datetime.now().isoformat(),
                    'val_size': len(self.dataset),
                    'val_start_idx': self.val_start_idx,
                    'dataset': 'Spanish-to-English'
                }
                
                # Update summaries (remove old entry if exists)
                checkpoint_summaries = [s for s in checkpoint_summaries if s['checkpoint'] != checkpoint_name]
                checkpoint_summaries.append(summary)
                
                self.save_checkpoint_summary(checkpoint_summaries, summary_file)
                
            except Exception as e:
                print(f"Error evaluating checkpoint {checkpoint_name}: {e}")
                error_summary = {
                    'checkpoint': checkpoint_name,
                    'checkpoint_type': checkpoint_type,
                    'error': str(e),
                    'failed_at': datetime.now().isoformat(),
                    'dataset': 'English-to-Spanish'
                }
                checkpoint_summaries.append(error_summary)
                self.save_checkpoint_summary(checkpoint_summaries, summary_file)
                
            finally:
                # Clean up model memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"Memory cleaned up after {checkpoint_name}")
        
        # Final summary
        print("\n" + "="*60 + "\nFINAL PERFORMANCE EVALUATION SUMMARY\n" + "="*60)
        successful_summaries = [s for s in checkpoint_summaries if 'error' not in s]
        
        if successful_summaries:
            print("Checkpoint Results:")
            for summary in successful_summaries:
                print(f"  {summary['checkpoint']} ({summary['checkpoint_type']}): BLEU {summary['bleu_score']:.4f}, ROUGE-L {summary['rougeL']:.4f}")
            
            # Group by checkpoint type for analysis
            type_summaries = {}
            for summary in successful_summaries:
                ctype = summary['checkpoint_type']
                if ctype not in type_summaries:
                    type_summaries[ctype] = {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
                type_summaries[ctype]['bleu'].append(summary['bleu_score'])
                type_summaries[ctype]['rouge1'].append(summary['rouge1'])
                type_summaries[ctype]['rouge2'].append(summary['rouge2'])
                type_summaries[ctype]['rougeL'].append(summary['rougeL'])
            
            print("\nAverage scores by checkpoint type:")
            for ctype, scores in type_summaries.items():
                avg_bleu = sum(scores['bleu']) / len(scores['bleu'])
                avg_rouge1 = sum(scores['rouge1']) / len(scores['rouge1'])
                avg_rouge2 = sum(scores['rouge2']) / len(scores['rouge2'])
                avg_rougeL = sum(scores['rougeL']) / len(scores['rougeL'])
                print(f"  {ctype}: BLEU {avg_bleu:.4f}, ROUGE-1 {avg_rouge1:.4f}, ROUGE-2 {avg_rouge2:.4f}, ROUGE-L {avg_rougeL:.4f} ({len(scores['bleu'])} checkpoints)")
            
            overall_avg_bleu = sum(s['bleu_score'] for s in successful_summaries) / len(successful_summaries)
            overall_avg_rougeL = sum(s['rougeL'] for s in successful_summaries) / len(successful_summaries)
            print(f"\nOverall averages: BLEU {overall_avg_bleu:.4f}, ROUGE-L {overall_avg_rougeL:.4f}")
            
            # Find best checkpoint by different metrics
            best_bleu = max(successful_summaries, key=lambda x: x['bleu_score'])
            best_rougeL = max(successful_summaries, key=lambda x: x['rougeL'])
            
            print(f"Best BLEU: {best_bleu['checkpoint']} (BLEU: {best_bleu['bleu_score']:.4f})")
            print(f"Best ROUGE-L: {best_rougeL['checkpoint']} (ROUGE-L: {best_rougeL['rougeL']:.4f})")
            
            if best_bleu['checkpoint'] == best_rougeL['checkpoint']:
                print(f"Same checkpoint achieves best performance on both metrics!")
            else:
                print(f"Different checkpoints excel at different metrics.")
        else:
            print("No successful evaluations completed.")
        
        return all_results, checkpoint_summaries


def main():
    """Example usage for unified performance evaluation"""
    # Configuration
    BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    MODELCODE = "llama-3.2-1b-it-codeUltraFeedback-lora-att-r16-lr1e-4-bs8"
    CHECKPOINT_DIR = f"models/{MODELCODE}"  # Directory containing mixed checkpoint types
    
    # Initialize evaluator
    evaluator = UnifiedPerformanceEvaluator(
        base_model_name=BASE_MODEL_NAME,
        val_size=1000,  # Use 1000 examples for validation
        val_start_idx=5000,  # Start after your training data
        random_seed=42
    )
    
    # Run evaluation
    results, summaries = evaluator.run_evaluation(
        checkpoint_dir=CHECKPOINT_DIR,
        output_file=f"results/performance/{MODELCODE}.csv",
        batch_size=2,  # Adjust based on available memory
        save_frequency=50,  # Save every 50 examples
        resume=True
    )
    
    print(f"\nUnified performance evaluation completed! Results saved.")


if __name__ == "__main__":
    main()