import pandas as pd
import argparse
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

def analyze_text(text):
    """Analyze text to extract various metadata features without relying on punkt_tab."""
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Calculate length features
    word_count = len(text.split())
    
    # Simple sentence count (approximate using punctuation)
    sentence_count = len(re.split(r'[.!?]+', text)) - 1
    if sentence_count < 1:  # Handle case with no sentence-ending punctuation
        sentence_count = 1
    
    # Determine content type
    content_types = []
    if re.search(r'(story|once upon|happened|experience|encounter)', text.lower()):
        content_types.append('story')
    if re.search(r'(step|process|framework|method|how to|guide|blueprint)', text.lower()):
        content_types.append('framework')
    if re.search(r'(example|instance|case|scenario)', text.lower()):
        content_types.append('example')
    if re.search(r'(problem|solution|challenge|overcome|difficulty)', text.lower()):
        content_types.append('problem_solution')
    if re.search(r'(email|subject line|preview|newsletter)', text.lower()):
        content_types.append('email_content')
    
    # Default if none found
    if not content_types:
        content_types.append('general')
    
    # Get sentiment scores
    sentiment = sia.polarity_scores(text)
    
    # Determine audience target
    audience = []
    if re.search(r'(parent|child|kid|family|mom|dad)', text.lower()):
        audience.append('parents')
    if re.search(r'(business|entrepreneur|company|startup|founder)', text.lower()):
        audience.append('business')
    if re.search(r'(student|learn|education|school|college)', text.lower()):
        audience.append('education')
    
    # Default audience
    if not audience:
        audience.append('general')
    
    # Check for calls to action
    has_cta = 1 if re.search(r'(click|subscribe|sign up|join|download|try|get started)', text.lower()) else 0
    
    # Extract topics
    topics = []
    if re.search(r'(stress|anxiety|worry|overwhelm)', text.lower()):
        topics.append('stress_management')
    if re.search(r'(capability|able|skill|learn|growth)', text.lower()):
        topics.append('capability_building')
    if re.search(r'(confidence|trust|believe|faith)', text.lower()):
        topics.append('confidence')
    if re.search(r'(judgment|critique|opinion|evaluate)', text.lower()):
        topics.append('judgment')
    
    # Default topic
    if not topics:
        topics.append('general')
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'content_type': "|".join(content_types),
        'sentiment_positive': sentiment['pos'],
        'sentiment_negative': sentiment['neg'],
        'sentiment_neutral': sentiment['neu'],
        'sentiment_compound': sentiment['compound'],
        'audience': "|".join(audience),
        'has_cta': has_cta,
        'topics': "|".join(topics),
        'length_category': 'short' if word_count < 200 else 'medium' if word_count < 500 else 'long'
    }

def add_metadata_to_transcript_csv(input_file, output_file=None):
    """Add metadata columns to transcript embedding CSV files."""
    if output_file is None:
        # Create output filename based on input filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_with_metadata{ext}"
    
    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if this is indeed a transcript embedding file
    if not all(col in df.columns for col in ['id', 'source', 'chunk_text']):
        raise ValueError("This doesn't appear to be a transcript embedding CSV file")
    
    print(f"Processing {len(df)} rows...")
    
    # Create metadata columns
    metadata = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}/{len(df)}")
        
        # Extract text features from chunk_text
        features = analyze_text(row['chunk_text'])
        metadata.append(features)
    
    # Convert list of dictionaries to DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Concatenate with original DataFrame
    result_df = pd.concat([df, metadata_df], axis=1)
    
    # Save to new file
    print(f"Saving enhanced file to: {output_file}")
    result_df.to_csv(output_file, index=False)
    
    print(f"Added metadata to {len(df)} rows")
    print(f"New columns: {', '.join(metadata_df.columns)}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metadata to transcript embedding CSV files")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", help="Path to save the enhanced CSV file (optional)")
    
    args = parser.parse_args()
    
    add_metadata_to_transcript_csv(args.input_file, args.output)