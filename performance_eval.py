import os
import re
import csv
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


class UnifiedMTPerformanceEvaluator:
    """
    Unified MT Performance evaluator that handles LoRA, QLoRA, and full fine-tuning checkpoints
    Evaluates translation quality using chrF metric only (lightweight version)
    """
    def __init__(self, 
                 base_model_name: str,
                 mt_dataset_name: str,
                 device: str = "auto", 
                 random_seed: int = 42):
        """
        Initialize the unified evaluator for MT performance.
        
        Args:
            base_model_name: The base model name or path for tokenizer and base model loading
            mt_dataset_name: HuggingFace dataset name for the filtered MT preference dataset
            device: Device to run inference on
            random_seed: Random seed for reproducible evaluation
        """
        self.base_model_name = base_model_name
        self.mt_dataset_name = mt_dataset_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        
        # Load tokenizer
        print(f"Loading tokenizer from: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load evaluation dataset (test split)
        self.load_dataset()
        
        # Initialize chrF metric
        self.init_chrf()

    def load_dataset(self):
        """Load the filtered MT preference dataset test split"""
        try:
            ds = load_dataset(self.mt_dataset_name)
            self.dataset = pd.DataFrame(ds['test'])
            print(f"Loaded MT performance dataset with {len(self.dataset)} test samples")
            
            required_columns = ['source_text', 'chosen', 'source_language']
            for col in required_columns:
                if col not in self.dataset.columns:
                    raise ValueError(f"MT dataset must contain '{col}' column")
            
            # Show language distribution
            lang_dist = self.dataset['source_language'].value_counts()
            print("Language distribution in test set:")
            for lang, count in lang_dist.items():
                print(f"  {lang}: {count}")
                
        except Exception as e:
            print(f"Error loading MT dataset: {e}")
            raise

    def init_chrf(self):
        """Initialize chrF metric"""
        try:
            print("Loading chrF metric...")
            self.chrf = evaluate.load("chrf")
            print("chrF metric initialized successfully")
            
        except Exception as e:
            print(f"Error initializing chrF metric: {e}")
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

    def generate_translation(self, model, source_text: str, source_language: str, max_length: int = 256) -> str:
        """Generate translation from model using the same format as training data"""
        try:
            # Create the same prompt format as in the original dataset
            prompt = f"Translate the following {source_language.title()} source text to English:\n{source_language.title()}: {source_text}\nEnglish:"
            
            # Format using Llama chat template
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
                    do_sample=False, 
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

    def extract_reference_translation(self, chosen_response: str) -> str:
        """Extract clean translation from chosen response (remove special tokens)"""
        # Remove assistant tokens and other formatting
        clean_translation = re.sub(r'<\|im_start\|>assistant\s*', '', chosen_response)
        clean_translation = re.sub(r'<\|im_end\|>.*$', '', clean_translation)
        return clean_translation.strip()

    def compute_chrf(self, predictions: List[str], references: List[str]) -> float:
        """Compute chrF score"""
        try:
            chrf_result = self.chrf.compute(predictions=predictions, references=references)
            return chrf_result['score']
        except Exception as e:
            print(f"Error computing chrF: {e}")
            return None

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

    def save_checkpoint_summary(self, checkpoint_summaries: List[Dict], summary_file: str):
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
            if 'error' not in str(result.get('model_translation', '')).lower():
                checkpoint_counts[checkpoint] += 1
        
        # Return checkpoints that have been fully evaluated
        total_questions = len(self.dataset)
        return {cp for cp, count in checkpoint_counts.items() if count >= total_questions}

    def evaluate_checkpoint(self, 
                          model, 
                          checkpoint_name: str, 
                          checkpoint_type: str,
                          batch_size: int = 8, 
                          output_file: str = "mt_performance_results.csv", 
                          save_frequency: int = 20) -> Tuple[List[Dict], float]:
        """Evaluate a single checkpoint on translation performance"""
        results = []
        all_predictions = []
        all_references = []
        
        print(f"Evaluating {checkpoint_type} checkpoint: {checkpoint_name}")
        
        # Check for existing results
        existing_results = self.load_existing_results(output_file)
        completed_sources = {r['source_text'] for r in existing_results if r['checkpoint'] == checkpoint_name}
        print(f"Found {len(completed_sources)} already completed translations for {checkpoint_name}")
        
        processed_count = 0
        
        for i in tqdm(range(0, len(self.dataset), batch_size), desc="Processing batches"):
            batch = self.dataset.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                source_text = row['source_text']
                source_language = row['source_language']
                chosen_response = row['chosen']
                
                if source_text in completed_sources:
                    continue
                
                try:
                    # Generate model translation
                    model_translation = self.generate_translation(model, source_text, source_language)
                    
                    # Extract reference translation from chosen response
                    reference_translation = self.extract_reference_translation(chosen_response)
                    
                    # Store for metric computation
                    all_predictions.append(model_translation)
                    all_references.append(reference_translation)
                    
                    # Create result record
                    result = {
                        'checkpoint': checkpoint_name,
                        'checkpoint_type': checkpoint_type,
                        'source_language': source_language,
                        'source_text': source_text,
                        'reference_translation': reference_translation,
                        'model_translation': model_translation,
                        'translation_length': len(model_translation),
                        'reference_length': len(reference_translation),
                        'timestamp': datetime.now().isoformat(),
                        'dataset_size': len(self.dataset),
                        'random_seed': self.random_seed
                    }
                    
                    results.append(result)
                    processed_count += 1
                    
                    # Save intermediate results
                    if processed_count % save_frequency == 0:
                        self.append_results(results[-save_frequency:], output_file)
                    
                except Exception as e:
                    print(f"Error processing translation for {checkpoint_name}: {e}")
                    error_result = {
                        'checkpoint': checkpoint_name,
                        'checkpoint_type': checkpoint_type,
                        'source_language': source_language,
                        'source_text': source_text,
                        'reference_translation': self.extract_reference_translation(chosen_response),
                        'model_translation': f"ERROR: {str(e)}",
                        'translation_length': 0,
                        'reference_length': len(self.extract_reference_translation(chosen_response)),
                        'timestamp': datetime.now().isoformat(),
                        'dataset_size': len(self.dataset),
                        'random_seed': self.random_seed
                    }
                    self.append_results([error_result], output_file)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save remaining results
        if results and (processed_count % save_frequency != 0):
            self.append_results(results[-(processed_count % save_frequency):], output_file)
        
        # Compute overall chrF score
        if all_predictions and all_references:
            chrf_score = self.compute_chrf(all_predictions, all_references)
        else:
            chrf_score = None
        
        print(f"Checkpoint {checkpoint_name} ({checkpoint_type}) - chrF: {chrf_score:.3f if chrf_score is not None else 'N/A'}")
        
        return results, chrf_score

    def run_evaluation(self, 
                      checkpoint_dir: str, 
                      output_file: str = "mt_performance_results.csv", 
                      batch_size: int = 8, 
                      save_frequency: int = 20, 
                      resume: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Run evaluation on all checkpoints"""
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
            
            print(f"\n{'='*60}\nStarting evaluation of {checkpoint_type} checkpoint: {checkpoint_name}\n{'='*60}")
            
            model = self.load_checkpoint(checkpoint_path, checkpoint_type)
            if model is None:
                continue
            
            try:
                results, chrf_score = self.evaluate_checkpoint(
                    model, checkpoint_name, checkpoint_type, batch_size, output_file, save_frequency
                )
                
                all_results.extend(results)
                
                # Create summary
                summary = {
                    'checkpoint': checkpoint_name,
                    'checkpoint_type': checkpoint_type,
                    'chrf_score': chrf_score,
                    'total_samples': len(results),
                    'completed_at': datetime.now().isoformat(),
                    'dataset_size': len(self.dataset),
                    'random_seed': self.random_seed,
                    'dataset': 'MT-Pref-Filtered'
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
                    'dataset': 'MT-Pref-Filtered'
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
        print("\n" + "="*60 + "\nFINAL MT PERFORMANCE EVALUATION SUMMARY\n" + "="*60)
        successful_summaries = [s for s in checkpoint_summaries if 'error' not in s]
        
        if successful_summaries:
            print("Checkpoint Results:")
            for summary in successful_summaries:
                chrf_str = f"{summary['chrf_score']:.3f}" if summary['chrf_score'] is not None else "N/A"
                print(f"  {summary['checkpoint']} ({summary['checkpoint_type']}): chrF={chrf_str}")
            
            # Group by checkpoint type for analysis
            type_summaries = {}
            for summary in successful_summaries:
                ctype = summary['checkpoint_type']
                if ctype not in type_summaries:
                    type_summaries[ctype] = []
                score = summary['chrf_score']
                if score is not None:
                    type_summaries[ctype].append(score)
            
            print("\nAverage chrF by checkpoint type:")
            for ctype, scores in type_summaries.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"  {ctype}: {avg_score:.3f} average ({len(scores)} checkpoints)")
            
            # Overall average
            all_scores = [s['chrf_score'] for s in successful_summaries if s['chrf_score'] is not None]
            if all_scores:
                overall_avg = sum(all_scores) / len(all_scores)
                print(f"\nOverall average chrF: {overall_avg:.3f}")
            
        else:
            print("No successful evaluations completed.")
        
        return all_results, checkpoint_summaries


def main():
    """Example usage for unified MT performance evaluation"""
    # Configuration
     # Configuration
    LR = 1-4
    BS = 4
    MODE = "dpo"
    BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_CODE = f"llama-3.2-1b-it-translation-{MODE}-lr{LR}-bs{BS}"    
    CHECKPOINT_DIR = f"models/{MODEL_CODE}"  # Directory containing mixed checkpoint types
    
    MT_DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"  # Your filtered dataset

    
    # Initialize evaluator
    evaluator = UnifiedMTPerformanceEvaluator(
        base_model_name=BASE_MODEL_NAME,
        mt_dataset_name=MT_DATASET_NAME,
        random_seed=42
    )
    
    # Run evaluation
    results, summaries = evaluator.run_evaluation(
        checkpoint_dir=CHECKPOINT_DIR,
        output_file=f"results/performance/{MODEL_CODE}.csv",
        batch_size=8,  # Increased since no COMET overhead
        save_frequency=20, 
        resume=True
    )
    
    print(f"\nMT Performance evaluation completed! Results saved.")


if __name__ == "__main__":
    main()