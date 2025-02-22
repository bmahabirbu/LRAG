import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import (
    hf_embed,
)
from lightrag.llm.openai import openai_complete_if_cache

from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

import argparse

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "dummy"
API_KEY = "dummy"
URL = "http://localhost:8080/v1"

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=URL,
        api_key=API_KEY,
        **kwargs,
    )

WORKING_DIR = "m"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    # llm_model_func=hf_model_complete,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=2000,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=2000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)
# Replace with docling

# Now indexing
with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

print(
    rag.query("is the story Iris well written?", param=QueryParam(mode="naive"))
)