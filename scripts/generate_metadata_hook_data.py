#!/usr/bin/env python3

import pandas as pd
import argparse
import re
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

def analyze_hook(hook_text, hook_type=None, platform=None):
    """Analyze hook text to extract various metadata features."""
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Handle non-string inputs
    if not isinstance(hook_text, str):
        hook_text = str(hook_text) if hook_text is not None else ""
    
    # Calculate length features
    word_count = len(hook_text.split())
    character_count = len(hook_text)
    
    # Determine hook style
    hook_styles = []
    if hook_text:
        if re.search(r'\?', hook_text):
            hook_styles.append('question')
        if re.search(r'(secret|hidden|unknown|discover|reveal)', hook_text.lower()):
            hook_styles.append('curiosity')
        if re.search(r'(never|always|every|nothing|everything|no one|everyone)', hook_text.lower()):
            hook_styles.append('absolute')
        if re.search(r'(you|your|you\'ll|you\'re|you\'ve)', hook_text.lower()):
            hook_styles.append('second_person')
        if re.search(r'(I|my|me|mine|we|our|us)', hook_text.lower()):
            hook_styles.append('first_person')
        if re.search(r'(how to|guide|steps|ways|tips)', hook_text.lower()):
            hook_styles.append('instructional')
        if re.search(r'(amazing|incredible|unbelievable|shocking|stunning)', hook_text.lower()):
            hook_styles.append('dramatic')
        if re.search(r'(\d+%|\d+ percent|double|triple|times|increased)', hook_text.lower()):
            hook_styles.append('statistic')
    
    # Default if none found
    if not hook_styles:
        hook_styles.append('general')
    
    # Get sentiment scores for text hooks
    sentiment = sia.polarity_scores(hook_text)
    
    # Determine hook strength based on sentiment intensity
    if abs(sentiment['compound']) > 0.5:
        hook_strength = 'strong'
    elif abs(sentiment['compound']) > 0.2:
        hook_strength = 'medium'
    else:
        hook_strength = 'mild'
    
    # Map hook length to a category
    if word_count == 0:
        length_category = 'non_text'
    elif word_count < 5:
        length_category = 'very_short'
    elif word_count < 10:
        length_category = 'short'
    elif word_count < 20:
        length_category = 'medium'
    else:
        length_category = 'long'
    
    # Check for emoji, numbers, and questions
    has_emoji = 1 if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', hook_text) else 0
    has_number = 1 if re.search(r'\d+', hook_text) else 0
    has_question = 1 if '?' in hook_text else 0
    
    return {
        'word_count': word_count,
        'character_count': character_count,
        'hook_style': "|".join(hook_styles),
        'sentiment_positive': sentiment['pos'],
        'sentiment_negative': sentiment['neg'],
        'sentiment_neutral': sentiment['neu'],
        'sentiment_compound': sentiment['compound'],
        'hook_strength': hook_strength,
        'length_category': length_category,
        'has_emoji': has_emoji,
        'has_number': has_number,
        'has_question': has_question
    }

def add_metadata_to_hook_csv(input_file, output_file=None):
    """Add metadata columns to hook database CSV files."""
    if output_file is None:
        # Create output filename based on input filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_with_metadata{ext}"
    
    # Read the CSV file with the correct delimiter
    print(f"Reading file: {input_file}")
    
    try:
        # Use the correct delimiter: semicolon
        df = pd.read_csv(
            input_file,
            delimiter=';',     # Use semicolon as the delimiter
            encoding='utf-8',  # Explicit encoding
            encoding_errors='replace',  # Replace encoding errors
            low_memory=False   # Don't use low_memory mode
        )
        print(f"Successfully loaded CSV with {len(df.columns)} columns and {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        raise
    
    print("Columns:", df.columns.tolist())
    
    # Determine the hook text column - might be 'hook_text', 'text', etc.
    text_column = None
    for col in ['hook_text', 'text', 'hook', 'content', 'Actual Spoken Hook']:
        if col in df.columns:
            text_column = col
            print(f"Using column '{text_column}' as hook text")
            break
    
    if not text_column:
        # If no obvious text column is found, use the first column that's not 'id' or 'embedding'
        for col in df.columns:
            if col.lower() not in ['id', 'embedding']:
                text_column = col
                print(f"Using column '{text_column}' as hook text")
                break
    
    if not text_column:
        raise ValueError("Could not identify a text column in this CSV file")
    
    # Create metadata columns
    metadata = []
    errors = 0
    print(f"Processing {len(df)} rows to generate metadata...")
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}/{len(df)}")
        
        try:
            # Get hook type and platform if they exist
            hook_type = row.get('Content Type', None) if 'Content Type' in df.columns else None
            platform = row.get('Niche', None) if 'Niche' in df.columns else None
            
            # Extract features from hook text
            features = analyze_hook(row[text_column], hook_type, platform)
            metadata.append(features)
        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")
            errors += 1
            # Add empty metadata for this row
            metadata.append({
                'word_count': 0,
                'character_count': 0,
                'hook_style': 'error',
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 0,
                'sentiment_compound': 0,
                'hook_strength': 'unknown',
                'length_category': 'unknown',
                'has_emoji': 0,
                'has_number': 0,
                'has_question': 0
            })
    
    if errors > 0:
        print(f"Warning: Encountered errors in {errors} rows")
    
    # Convert list of dictionaries to DataFrame
    metadata_df = pd.DataFrame(metadata)
    print(f"Generated metadata columns: {metadata_df.columns.tolist()}")
    
    # Concatenate with original DataFrame
    result_df = pd.concat([df, metadata_df], axis=1)
    
    # Save to new file - use the same delimiter as the input
    print(f"Saving enhanced file to: {output_file}")
    result_df.to_csv(output_file, index=False, sep=';')
    
    print(f"Added metadata to {len(df)} rows")
    print(f"New columns: {', '.join(metadata_df.columns)}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metadata to hook database CSV files")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", help="Path to save the enhanced CSV file (optional)")
    
    args = parser.parse_args()
    
    add_metadata_to_hook_csv(args.input_file, args.output)