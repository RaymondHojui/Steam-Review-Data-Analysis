# Copyright (C) 2026 Raymond Ho-Jui
#
# This file is part of Steam Review Data Analysis.
# Licensed under the GNU Affero General Public License v3.0.
# See the LICENSE file in the repository root.

import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt

csv_path = r"data/utf_final_result.csv"
df = pd.read_csv(csv_path)

recommend_col = df["recommend"]
recommend_col = recommend_col.astype(str)
recommend_col = recommend_col.str.strip()

is_negative = recommend_col == "Not Recommended"


neg_df = df[is_negative]

def parse_labels_cell(cell): 
    text = str(cell).strip()
    if text == "":
        return []
    labels = ast.literal_eval(text)
    if type(labels) == list:
        return labels
    else:
        return []

all_neg_tags = []

col = neg_df["llm_labels"]
for cell in col:
    labels_list = parse_labels_cell(cell)
    for tag in labels_list:
        all_neg_tags.append(tag)

cleaned_tags = []
for tag in all_neg_tags:
    if type(tag) == str:
        t = tag.strip().lower()
        if t != "":
            cleaned_tags.append(t)

tag_counts = Counter(cleaned_tags)

top_n = 20
most_common = tag_counts.most_common(top_n)

tags = []
counts = []
for tag, count in most_common:
    tags.append(tag)
    counts.append(count)

plt.figure(figsize=(12, 6))
plt.bar(tags, counts)
plt.xlabel("Tag")
plt.ylabel("Frequency")
plt.title("Top 20 Tags in Negative (Not Recommended) Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
