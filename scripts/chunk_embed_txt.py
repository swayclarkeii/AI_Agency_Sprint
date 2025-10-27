"""
Script for chunking text files with strict size control and generating embeddings.

This script offers two chunking modes:
1. NEWSLETTER: Larger chunks (~700 words) suitable for newsletter generation
2. VIDEO: Smaller chunks (~300 words) suitable for short video script generation
"""

import re
import argparse
import pandas as pd
import time
import tiktoken
import nltk
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Define paths directly
RAW_ROOT = Path("data/raw")
CHUNKS = Path("data/chunks")
KB = Path("data/kb")

# Define chunking modes
MODES = {
    "newsletter": {
        "target_words": 700,
        "max_words": 770,  # 10% tolerance
        "min_words": 300,  # Don't create tiny chunks
        "sentence_overlap": 3,  # Number of sentences to overlap
        "description": "Larger chunks (~700 words) suitable for newsletter generation"
    },
    "video": {
        "target_words": 300, 
        "max_words": 330,  # 10% tolerance
        "min_words": 150,  # Don't create tiny chunks
        "sentence_overlap": 2,  # Number of sentences to overlap
        "description": "Smaller chunks (~300 words) suitable for short video script generation"
    }
}

load_dotenv()
client = OpenAI()

# Install dependencies
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except:
    import subprocess
    print("Installing nltk and downloading punkt tokenizer...")
    subprocess.check_call(["pip", "install", "nltk"])
    import nltk
    nltk.download('punkt')

def count_tokens(text, model="text-embedding-3-small"):
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: estimate tokens as ~4 characters per token
        return len(text) // 4

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Replace multiple newlines with paragraph markers
    text = re.sub(r"\n\s*\n", " <PARA> ", text)
    
    # Replace single newlines with spaces
    text = text.replace("\n", " ")
    
    # Simple regex for sentence splitting
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # Further split on common sentence endings
    result = []
    for sent in sentences:
        # Split on periods, exclamation points, and question marks
        parts = re.split(r'([.!?])', sent)
        for i in range(0, len(parts) - 1, 2):
            if parts[i].strip():
                result.append(parts[i].strip() + parts[i+1])
    
    # Process paragraph markers
    final_result = []
    for sent in result:
        if "<PARA>" in sent:
            parts = sent.split("<PARA>")
            for i, part in enumerate(parts):
                if part.strip():
                    final_result.append(part.strip())
                if i < len(parts) - 1:
                    final_result.append("<PARA>")
        else:
            final_result.append(sent)
    
    # Handle empty result case
    if not final_result and text.strip():
        return [text.strip()]
    
    return final_result

def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())

def chunk_text_strict(
    text: str, 
    mode: str = "newsletter",
    max_tokens: int = 7500
) -> List[str]:
    """
    Split text into chunks with strict size control, respecting sentence boundaries.
    
    Args:
        text: The input text to chunk
        mode: Chunking mode - "newsletter" or "video"
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Get mode parameters
    params = MODES.get(mode.lower(), MODES["newsletter"])
    target_words = params["target_words"]
    max_words = params["max_words"]
    min_words = params["min_words"]
    sentence_overlap = params["sentence_overlap"]
    
    print(f"Using {mode} mode: target {target_words} words, max {max_words} words, {sentence_overlap} sentence overlap")
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    print(f"Split text into {len(sentences)} sentences")
    
    # Calculate word counts for each sentence
    sentence_word_counts = [count_words(sent) for sent in sentences]
    
    # Initialize chunks
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    # Process sentences
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        word_count = sentence_word_counts[i]
        
        # Special handling for paragraph markers
        if sentence == "<PARA>":
            if current_chunk:
                current_chunk.append("")  # Add an empty line for paragraph break
            i += 1
            continue
        
        # Handle oversized single sentences
        if word_count > max_words:
            if current_chunk:
                # Finish current chunk first
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split the long sentence into parts
            words = sentence.split()
            for j in range(0, len(words), max_words):
                part = words[j:min(j+max_words, len(words))]
                part_text = " ".join(part)
                
                # If this isn't the first part and it's very small, combine with previous
                if j > 0 and len(part) < min_words and chunks:
                    chunks[-1] = chunks[-1] + " " + part_text
                else:
                    chunks.append(part_text)
            
            i += 1
            continue
        
        # Check if adding this sentence would exceed max words
        if current_word_count + word_count > max_words and current_word_count >= min_words:
            # Save current chunk and start a new one with overlap
            chunks.append(" ".join(current_chunk))
            
            # Calculate overlap
            overlap_start = max(0, len(current_chunk) - sentence_overlap)
            current_chunk = current_chunk[overlap_start:]
            current_word_count = sum(sentence_word_counts[i - len(current_chunk):i])
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += word_count
        i += 1
    
    # Add the last chunk if it's not empty
    if current_chunk and current_word_count >= min_words:
        chunks.append(" ".join(current_chunk))
    
    # Final token check to ensure no chunk exceeds token limit
    result_chunks = []
    for chunk in chunks:
        if count_tokens(chunk) <= max_tokens:
            result_chunks.append(chunk)
        else:
            print(f"Warning: Chunk exceeds token limit. Splitting further...")
            # Recursively split this chunk with more aggressive parameters
            recursive_mode = "video" if mode.lower() == "newsletter" else mode.lower()
            recursive_chunks = chunk_text_strict(
                chunk,
                mode=recursive_mode,
                max_tokens=max_tokens
            )
            result_chunks.extend(recursive_chunks)
    
    # Verify chunk sizes
    word_counts = [count_words(chunk) for chunk in result_chunks]
    print(f"Created {len(result_chunks)} chunks")
    print(f"Average chunk size: {sum(word_counts)/len(word_counts):.1f} words")
    print(f"Min chunk size: {min(word_counts)} words")
    print(f"Max chunk size: {max(word_counts)} words")
    
    return result_chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def generate_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate an embedding for a text using OpenAI's API with retry logic."""
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 10) -> List[List[float]]:
    """Generate embeddings for a list of texts with progress reporting."""
    total = len(texts)
    embeddings = []
    
    print(f"Generating embeddings for {total} chunks using {model}...")
    
    for i, text in enumerate(texts):
        if i > 0 and i % batch_size == 0:
            print(f"Progress: {i}/{total} chunks embedded ({i/total*100:.1f}%)")
            time.sleep(0.1)
        
        try:
            embedding = generate_embedding(text, model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to embed chunk {i}: {str(e)}")
            embeddings.append([0.0] * 1536)
    
    print(f"Completed embedding {len(embeddings)}/{total} chunks")
    return embeddings

def get_all_txt_files(folder: Path) -> List[Path]:
    """Get all .txt files in a folder."""
    if not folder.exists():
        return []
    return sorted(folder.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)

def process_file(
    file_path: Path, 
    mode: str = "newsletter",
    max_tokens: int = 7500,
    source_tag: str = "", 
    embedding_model: str = "text-embedding-3-small"
) -> pd.DataFrame:
    """Process a single text file: chunk it and generate embeddings."""
    print(f"Processing file: {file_path.name}")
    
    # Read the text content
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    
    # Generate chunks with strict size control
    chunks = chunk_text_strict(text, mode=mode, max_tokens=max_tokens)
    
    # Create DataFrame with IDs and metadata
    chunk_ids = [f"{file_path.stem}_{mode}_chunk_{i+1:05d}" for i in range(len(chunks))]
    
    df = pd.DataFrame({
        "id": chunk_ids,
        "source": file_path.name,
        "mode": mode,
        "chunk_text": chunks,
    })
    
    # Add metadata/tags
    if source_tag:
        df["source_tag"] = source_tag
    
    # Add word counts for verification
    df["word_count"] = df["chunk_text"].apply(count_words)
    
    # Create match_text column (text + metadata)
    df["match_text"] = df.apply(
        lambda row: f"{row['chunk_text']} | source:{row['source']} | mode:{mode}", axis=1
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    df["embedding"] = embed_texts(df["match_text"].tolist(), model=embedding_model)
    
    # Save intermediate chunks
    chunks_dir = CHUNKS / "transcripts" / mode
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_csv = chunks_dir / f"{file_path.stem}_chunks.csv"
    
    # Save a version without the embeddings
    chunks_df = df[["id", "source", "mode", "word_count", "chunk_text"]].copy()
    chunks_df.to_csv(chunks_csv, index=False)
    print(f"→ Saved intermediate chunks to: {chunks_csv}")
    
    return df

def combine_into_kb(dataframes: List[pd.DataFrame], kb_path: Path, mode: str) -> pd.DataFrame:
    """Combine multiple dataframes into a single KB file, handling existing files."""
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine new dataframes
    combined_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    
    # Append to existing KB if it exists
    if kb_path.exists():
        try:
            old_kb = pd.read_csv(kb_path)
            
            # Get list of sources and mode in the new data
            sources_and_mode = set()
            for _, row in combined_df.iterrows():
                if "source" in row and "mode" in row:
                    sources_and_mode.add((row["source"], row["mode"]))
            
            # Filter out any existing chunks with same source and mode
            if "source" in old_kb.columns and "mode" in old_kb.columns:
                filter_condition = ~old_kb.apply(
                    lambda row: (row["source"], row["mode"]) in sources_and_mode, 
                    axis=1
                )
                old_kb = old_kb[filter_condition]
                print(f"Removed existing chunks from {len(sources_and_mode)} source/mode combinations in KB")
            
            # Combine with old KB
            final_kb = pd.concat([old_kb, combined_df], ignore_index=True)
            print(f"→ Appended {len(combined_df)} new chunks to existing KB ({len(old_kb)} previous chunks)")
            
        except Exception as e:
            print(f"Error reading existing KB file: {str(e)}")
            print("Creating new KB file instead...")
            final_kb = combined_df
    else:
        final_kb = combined_df
    
    # Save final KB
    final_kb.to_csv(kb_path, index=False)
    print(f"→ Updated KB at: {kb_path} (total: {len(final_kb)} chunks)")
    
    return final_kb

if __name__ == "__main__":
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Chunk text files with strict size control and generate embeddings")
    ap.add_argument("--dir", default="data/raw/video_scripts/text", 
                    help="Directory containing .txt files (default: data/raw/video_scripts/text)")
    ap.add_argument("--mode", choices=["newsletter", "video"], default="newsletter",
                    help="Chunking mode: newsletter (~700 words) or video (~300 words)")
    ap.add_argument("--max-tokens", type=int, default=7500, 
                    help="Maximum tokens per chunk (default: 7500)")
    ap.add_argument("--tag", default="", 
                    help="Optional source tag to add to metadata")
    ap.add_argument("--model", default="text-embedding-3-small", 
                    help="Embedding model to use")
    ap.add_argument("--clean", action="store_true", 
                    help="Clean text by removing timestamps before processing")
    args = ap.parse_args()
    
    # Ensure directories exist
    for directory in [RAW_ROOT, CHUNKS, KB]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get all text files in the directory
    transcripts_dir = Path(args.dir)
    txt_files = get_all_txt_files(transcripts_dir)
    
    print(f"=== Processing {len(txt_files)} text files from {transcripts_dir} in {args.mode} mode ===")
    
    if not txt_files:
        print(f"Error: No .txt files found in {transcripts_dir}")
        print("Tip: Convert other formats to TXT first or check the directory path")
        exit(1)
    
    # If clean flag is set, attempt to clean texts first
    if args.clean:
        try:
            from subprocess import run
            print("Cleaning text (removing timestamps)...")
            clean_script = "scripts/clean_transcript.py"
            clean_output_dir = f"data/cleaned/video_scripts/text/{args.mode}"
            Path(clean_output_dir).mkdir(parents=True, exist_ok=True)
            
            run(["python3", clean_script, 
                 "--in", str(transcripts_dir), 
                 "--out", clean_output_dir, 
                 "--drop-fillers"])
            
            # Use cleaned files instead
            transcripts_dir = Path(clean_output_dir)
            txt_files = get_all_txt_files(transcripts_dir)
            print(f"Using {len(txt_files)} cleaned text files from {transcripts_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean texts: {str(e)}")
            print("Proceeding with original files...")
    
    # Process each file
    all_dataframes = []
    for i, file_path in enumerate(txt_files):
        print(f"\n[{i+1}/{len(txt_files)}] Processing {file_path.name}...")
        try:
            df = process_file(
                file_path,
                mode=args.mode,
                max_tokens=args.max_tokens,
                source_tag=args.tag,
                embedding_model=args.model
            )
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Combine all into KB
    if all_dataframes:
        kb_path = KB / "transcripts" / f"transcripts_{args.mode}_embeddings.csv"
        final_kb = combine_into_kb(all_dataframes, kb_path, args.mode)
        
        # Show word count statistics
        word_counts = final_kb["word_count"].tolist()
        print(f"\n=== Chunk Size Statistics ===")
        print(f"Average: {sum(word_counts)/len(word_counts):.1f} words")
        print(f"Minimum: {min(word_counts)} words")
        print(f"Maximum: {max(word_counts)} words")
        
        print(f"\n=== Summary ===")
        print(f"Mode: {args.mode}")
        print(f"Processed {len(txt_files)} files")
        print(f"Created {sum(len(df) for df in all_dataframes)} chunks")
        print(f"Final KB has {len(final_kb)} total chunks")
        print(f"KB saved to: {kb_path}")
    else:
        print("No files were successfully processed.")