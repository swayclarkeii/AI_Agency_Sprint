import os

# Base directory to start searching
base_dir = 'data'

# Look for .rtf files
found_files = []
for root, dirs, files in os.walk(base_dir):
    rtf_files = [f for f in files if f.endswith('.rtf')]
    if rtf_files:
        print(f"Found {len(rtf_files)} RTF files in: {root}")
        for rtf_file in rtf_files[:3]:  # Show first 3 as examples
            found_files.append(os.path.join(root, rtf_file))
            print(f"  - {rtf_file}")
        if len(rtf_files) > 3:
            print(f"  - ... and {len(rtf_files)-3} more")

if not found_files:
    print("No .rtf files found in the 'data' directory tree.")
else:
    print(f"\nTotal .rtf files found: {len(found_files)}")
    print("\nSample commands to process these files:")
    sample_dir = os.path.dirname(found_files[0])
    print(f"\n# Convert RTFs to TXTs:")
    print(f"python3 convert_rtf.py --input \"{sample_dir}\" --output \"data/raw/video_scripts_txt\"")
    print(f"\n# Then clean the resulting TXT files:")
    print(f"python3 scripts/clean_transcript.py --in \"data/raw/video_scripts_txt\" --out \"data/cleaned/video_scripts/text\" --drop-fillers")
