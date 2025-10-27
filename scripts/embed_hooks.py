"""
Embedding generator for hook texts using OpenAI API.

This script processes a CSV file containing hook texts, generates embeddings for each hook
using OpenAI's embedding API, and saves the results to a new CSV file.
"""

import argparse
import os
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Import paths from common module
from common_paths import RAW_HOOKS, KB, latest_file

# Load environment variables and initialize client
load_dotenv()
client = OpenAI()

# Define embedding model as a constant
EMBEDDING_MODEL = "text-embedding-3-small"
REQUIRED_COLUMNS = ["hook_text"]
COLUMN_ALIASES = {
    "hook_text": [
        "hook_text",
        "Hook Text",
        "Hook",
        "Actual Spoken Hook",
    ],
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def create_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate an embedding for a single text using the OpenAI API.
    
    Args:
        text: The input text to embed
        model: Embedding model to use
        
    Returns:
        A list of floats representing the embedding vector
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def embed_texts(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings for a list of texts, with optional batching.
    
    Args:
        texts: List of texts to embed
        model: Embedding model to use
        batch_size: Number of texts to process in a single batch
        
    Returns:
        List of embedding vectors
    """
    # For simplicity, we'll process one text at a time
    # In a production environment, you'd want to batch these
    embeddings = []
    
    print(f"Generating embeddings for {len(texts)} texts using {model}...")
    
    for i, text in enumerate(texts):
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(texts)} texts")
        
        try:
            embedding = create_embedding(text, model=model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding text {i}: {str(e)}")
            # Add a placeholder or None to maintain alignment
            embeddings.append(None)
    
    return embeddings

def create_match_text(row: pd.Series) -> str:
    """
    Create a match text that captures the meaning of a hook.
    
    Args:
        row: A pandas Series representing a row in the hooks dataframe
        
    Returns:
        A string combining hook text with metadata
    """
    creator = row.get("creator", "")
    category = row.get("category", "")
    return f"{row['hook_text']} | creator:{creator} | category:{category}"

def normalize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column variants to expected internal names."""
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for hook texts")
    parser.add_argument("--csv", help="Path to hooks CSV (default: latest in data/raw/hooks)")
    parser.add_argument("--model", default=EMBEDDING_MODEL, 
                       help=f"OpenAI embedding model to use (default: {EMBEDDING_MODEL})")
    args = parser.parse_args()
    
    # Find source file
    src = args.csv or latest_file(RAW_HOOKS, "csv")
    assert src is not None, "No hooks CSV found. Please specify a file with --csv."
    print(f"Embedding hooks from: {src}")
    
    # Read and process data
    df = pd.read_csv(src)
    df = normalize_required_columns(df)
    
    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Create match text
    df["match_text"] = df.apply(create_match_text, axis=1)
    
    # Generate embeddings
    model = args.model
    df["embedding"] = embed_texts(df["match_text"].tolist(), model=model)
    
    # Save results
    out = KB / "hooks_with_embeddings.csv"
    KB.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"→ Wrote {len(df)} hooks with embeddings to {out}")

if __name__ == "__main__":
    main()
