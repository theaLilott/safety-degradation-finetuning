import re
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict
import random
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

def extract_translation_info(prompt: str) -> Tuple[str, str]:
    """
    Extract source language and source text from MT-Pref prompt format.
    
    Returns:
        Tuple of (source_language, source_text)
    """
    # Pattern to match the translation prompt format
    pattern = r'Translate the following (\w+) source text to English:\s*\1:\s*(.*?)\s*English:\s*<\|im_end\|>'
    
    match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
    
    if match:
        source_lang = match.group(1).lower()
        source_text = match.group(2).strip()
        return source_lang, source_text
    else:
        return None, None

def is_latin_script_language(language: str) -> bool:
    """
    Check if a language uses Latin script.
    """
    latin_script_languages = {
        'german', 'spanish', 'french', 'portuguese', 'italian', 'dutch',
        'english', 'catalan', 'galician', 'romanian', 'polish', 'czech',
        'slovak', 'slovenian', 'croatian', 'bosnian', 'serbian', 'albanian',
        'latvian', 'lithuanian', 'estonian', 'finnish', 'hungarian', 'danish',
        'swedish', 'norwegian', 'icelandic'
    }
    return language.lower() in latin_script_languages

def filter_and_process_dataset(dataset_name: str = "sardinelab/MT-pref") -> Tuple[List[Dict], Dict[str, int]]:
    """
    Filter MT-Pref dataset for Latin script -> English translations and process.
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # Work with the train split
    data = dataset['train']
    
    processed_data = []
    language_stats = defaultdict(int)
    
    print("Processing examples...")
    for idx, example in enumerate(data):
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(data)} examples...")
        
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Extract translation information from prompt
        source_lang, source_text = extract_translation_info(prompt)
        
        if source_lang is None:
            continue
            
        # Check if it's a Latin script language -> English translation
        if not is_latin_script_language(source_lang):
            continue
            
        processed_example = {
            'source_language': source_lang,
            'source_text': source_text,
            'chosen': chosen,
            'rejected': rejected,
            'prompt': prompt
        }
        
        processed_data.append(processed_example)
        language_stats[source_lang] += 1
    
    print(f"\nFiltered dataset statistics:")
    print(f"Total examples: {len(processed_data)}")
    print("Language distribution:")
    for lang, count in sorted(language_stats.items()):
        print(f"  {lang}: {count}")
    
    return processed_data, language_stats

def create_train_test_splits(processed_data: List[Dict], 
                           language_stats: Dict[str, int],
                           test_ratio: float = 0.167,  # ~1k out of 6k
                           random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Create proportional train/test splits maintaining language balance.
    """
    random.seed(random_seed)
    
    # Group examples by language
    examples_by_lang = defaultdict(list)
    for example in processed_data:
        examples_by_lang[example['source_language']].append(example)
    
    train_examples = []
    test_examples = []
    
    print(f"\nCreating train/test splits (test ratio: {test_ratio:.1%}):")
    
    for lang, examples in examples_by_lang.items():
        # Shuffle examples for this language
        random.shuffle(examples)
        
        # Calculate split point
        n_test = max(1, int(len(examples) * test_ratio))
        n_train = len(examples) - n_test
        
        # Split
        test_lang = examples[:n_test]
        train_lang = examples[n_test:]
        
        train_examples.extend(train_lang)
        test_examples.extend(test_lang)
        
        print(f"  {lang}: {n_train} train, {n_test} test")
    
    # Shuffle the final datasets to blend languages
    random.shuffle(train_examples)
    random.shuffle(test_examples)
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Test: {len(test_examples)}")
    
    return train_examples, test_examples

def create_huggingface_dataset(train_data: List[Dict], 
                              test_data: List[Dict]) -> DatasetDict:
    """
    Create HuggingFace DatasetDict from processed data.
    """
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def push_to_hub(dataset: DatasetDict, 
                repo_name: str,
                private: bool = True):
    """
    Push the processed dataset to HuggingFace Hub.
    """
    print(f"Pushing dataset to {repo_name}...")
    dataset.push_to_hub(repo_name, private=private, token= os.getenv("HF_TOKEN"))
    print("Dataset pushed successfully!")

def main():
    # Configuration
    OUTPUT_REPO = "safe-llm-finetune/mt-pref-latin-to-english"  # Change this!
    TEST_RATIO = 0.167  # ~1k out of 6k
    RANDOM_SEED = 42
    
    # Process dataset
    processed_data, language_stats = filter_and_process_dataset()
    
    # Create splits
    train_data, test_data = create_train_test_splits(
        processed_data, 
        language_stats, 
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )
    
    # Create HuggingFace dataset
    hf_dataset = create_huggingface_dataset(train_data, test_data)
    
    # Print sample
    print("\nSample from train set:")
    sample = hf_dataset['train'][0]
    print(f"Source language: {sample['source_language']}")
    print(f"Source text: {sample['source_text'][:100]}...")
    print(f"Chosen: {sample['chosen'][:100]}...")
    print(f"Rejected: {sample['rejected'][:100]}...")
    

    
    # Push to hub 
    push_to_hub(hf_dataset, OUTPUT_REPO, private=True)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()