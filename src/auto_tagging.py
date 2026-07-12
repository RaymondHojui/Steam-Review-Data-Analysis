# Copyright (C) 2026 Raymond Ho-Jui
#
# This file is part of Steam Review Data Analysis.
# Licensed under the GNU Affero General Public License v3.0.
# See the LICENSE file in the repository root.

import pandas as pd
from ollama import chat, ChatResponse
import time
import re

csv_path = r"data/utf_final_result.csv"
df = pd.read_csv(csv_path)
labels = []

total = len(df["review"])

for idx, review in enumerate(df["review"]):
    prompt = f'Please summarize 1-5 main topic tags for the following Steam review (e.g., monetization, story, performance, suggestions, online, bug, experience, complaints....). Output tags in a Python list format like ["xx","yy"]. Output tags only.\nReview: "{review}"\nTags:'
    try:
        response: ChatResponse = chat(
            model='deepseek-r1:14b', # you should change this to the modle your using
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        label = response['message']['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        label = "error"
    labels.append(label)
    print(f"[{idx + 1}/{total}] Tags: {label}")   # Show progress
    
# Add tags to DataFrame and save
df["llm_labels"] = labels
df.to_csv("reviews_with_llm_labels.csv", index=False, encoding="utf-8-sig")
print(df[["review", "llm_labels"]].head())
