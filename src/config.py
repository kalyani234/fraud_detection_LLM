"""
src/config.py - Configuration for LLM and API settings
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fast and cost-effective
    # model="gpt-4o",     # Alternative: stronger results but more expensive
    temperature=0.3,      # Lower = more deterministic
    max_tokens=1500,      # Limit response length
    api_key=api_key,
)



