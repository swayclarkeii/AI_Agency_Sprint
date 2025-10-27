import os
import csv
import statistics

# Path to your embeddings file
kb_file = "data/kb/transcripts/transcripts_with_embeddings.csv"

# Check if file exists
if not os.path.exists(kb_file):
    print(f"Error: File {kb_file} does not exist")
    # Look for other CSV files
    possible_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".csv"):
                possible_files.append(os.path.join(root, file))
    if possible_files:
        print(f"Found these CSV files instead:")
        for f in possible_files:
            print(f"  - {f}")
    exit(1)

# Read the CSV file
print(f"Reading file: {kb_file}")
with open(kb_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Get column headers
    
    # Print column headers
    print(f"Columns: {headers}")
    
    # Read all rows into memory
    rows = list(reader)
    
print(f"Total chunks: {len(rows)}")

# Check column indices
source_idx = headers.index('source') if 'source' in headers else None
chunk_idx = headers.index('chunk_text') if 'chunk_text' in headers else None
embedding_idx = headers.index('embedding') if 'embedding' in headers else None

# Analyze sources if available
if source_idx is not None:
    sources = {}
    for row in rows:
        source = row[source_idx]
        sources[source] = sources.get(source, 0) + 1
    
    print("\nSources and chunk counts:")
    for source, count in sources.items():
        print(f"  - {source}: {count} chunks")

# Analyze chunk lengths if available
if chunk_idx is not None:
    word_counts = []
    for row in rows:
        text = row[chunk_idx]
        word_count = len(text.split())
        word_counts.append(word_count)
    
    if word_counts:
        avg_words = statistics.mean(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        median_words = statistics.median(word_counts)
        
        print("\nChunk statistics:")
        print(f"  - Average words per chunk: {avg_words:.1f}")
        print(f"  - Median words per chunk: {median_words:.1f}")
        print(f"  - Min words per chunk: {min_words}")
        print(f"  - Max words per chunk: {max_words}")
        
        # Histogram of chunk sizes
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
        histogram = {f"{bins[i]}-{bins[i+1]-1}": 0 for i in range(len(bins)-1)}
        histogram[f"{bins[-2]}+"] = 0  # For the last bin
        
        for count in word_counts:
            for i in range(len(bins)-1):
                if bins[i] <= count < bins[i+1] or (i == len(bins)-2 and count >= bins[i]):
                    key = f"{bins[i]}-{bins[i+1]-1}" if i < len(bins)-2 else f"{bins[-2]}+"
                    histogram[key] += 1
                    break
        
        print("\nChunk size distribution:")
        for size_range, count in histogram.items():
            if count > 0:
                percentage = (count / len(word_counts)) * 100
                print(f"  - {size_range} words: {count} chunks ({percentage:.1f}%)")

# Check embedding format if available
if embedding_idx is not None and rows:
    sample_embedding = rows[0][embedding_idx]
    print(f"\nEmbedding format:")
    print(f"  - First few characters: {sample_embedding[:100]}...")
    
    # Try to determine dimensionality
    try:
        # Check if it's a string representation of a list
        if sample_embedding.startswith('[') and sample_embedding.endswith(']'):
            import ast
            embedding_vector = ast.literal_eval(sample_embedding)
            print(f"  - Embedding dimensions: {len(embedding_vector)}")
    except:
        print("  - Unable to determine embedding dimensions")