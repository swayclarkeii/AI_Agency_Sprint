import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple smoketest: list models (if permissions allow) or just print key status
print("Key loaded?", bool(openai.api_key))
