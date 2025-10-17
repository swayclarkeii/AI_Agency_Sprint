import os, openai, glob
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

for file in glob.glob("../outputs/day2_prompts/*.txt"):
    with open(file) as f:
        prompt = f.read()
    print(f"\nTesting {file}")
    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=200
    )
    output = response.choices[0].message.content
    outpath = file.replace(".txt","_result.txt")
    with open(outpath,"w") as o: o.write(output)
    print("Saved →", outpath)