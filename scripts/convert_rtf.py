from pathlib import Path
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Convert RTF files to plain text")
parser.add_argument("--input", required=True, help="Directory containing RTF files")
parser.add_argument("--output", required=True, help="Output directory for TXT files")
args = parser.parse_args()

# Define paths from arguments
input_dir = Path(args.input)
output_dir = Path(args.output)

# Ensure input directory exists
if not input_dir.exists():
    print(f"Error: Input directory does not exist: {input_dir}")
    exit(1)

# Create output directory if needed
output_dir.mkdir(parents=True, exist_ok=True)

# Import RTF converter
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    print("Installing striprtf package...")
    import subprocess
    subprocess.check_call(["pip", "install", "striprtf"])
    from striprtf.striprtf import rtf_to_text

# Count files
rtf_files = list(input_dir.glob("*.rtf"))
print(f"Found {len(rtf_files)} RTF files to convert in {input_dir}")

if not rtf_files:
    print("No RTF files found. Please check the input directory path.")
    exit(1)

# Process each RTF file
for rtf_file in rtf_files:
    print(f"Converting: {rtf_file.name}")
    try:
        with open(rtf_file, "r", encoding='utf-8', errors="ignore") as f:
            rtf_text = f.read()
        
        plain_text = rtf_to_text(rtf_text)
        output_file = output_dir / f"{rtf_file.stem}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(plain_text)
        
        print(f"✓ Created {output_file}")
    except Exception as e:
        print(f"Error processing {rtf_file}: {str(e)}")

print("Conversion complete!")
