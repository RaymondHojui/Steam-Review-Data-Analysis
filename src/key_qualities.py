# Copyright (C) 2026 Raymond Ho-Jui
#
# This file is part of Steam Review Data Analysis.
# Licensed under the GNU Affero General Public License v3.0.
# See the LICENSE file in the repository root.

import pandas as pd
import ast
import re

df = pd.read_csv("utf_final_result.csv")

df = df[df["recommend"].isin(["Recommended", "Not Recommended"])].copy()

def parse_labels(cell):
    try:
        labels = ast.literal_eval(str(cell))
        if isinstance(labels, list):
            return [
                re.sub(r"\s+", " ", str(label).strip().lower())
                for label in labels
                if str(label).strip()
            ]
        return []
    except:
        return []

df["parsed_labels"] = df["llm_labels"].apply(parse_labels)

tag_rows = df.explode("parsed_labels").rename(columns={"parsed_labels": "tag"})

tag_summary = (tag_rows.groupby(["tag", "recommend"]).size().unstack(fill_value=0).reset_index())

tag_summary.columns.name = None

tag_summary["total_reviews_with_tag"] = (tag_summary["Recommended"] + tag_summary["Not Recommended"])

tag_summary["recommended_rate"] = (tag_summary["Recommended"] / tag_summary["total_reviews_with_tag"])

tag_summary["not_recommended_rate"] = (tag_summary["Not Recommended"] / tag_summary["total_reviews_with_tag"])

min_count = 5
tag_summary = tag_summary[tag_summary["total_reviews_with_tag"] >= min_count].copy()


top_negative = tag_summary.sort_values(by="not_recommended_rate", ascending=False).head(3)


top_positive = tag_summary.sort_values(by="recommended_rate", ascending=False).head(3)

top_negative["not_recommended_rate"] = top_negative["not_recommended_rate"].apply(lambda x: f"{x:.1%}")
top_positive["recommended_rate"] = top_positive["recommended_rate"].apply(lambda x: f"{x:.1%}")

print("Top 3 Negative Tags")
print(top_negative[[
    "tag",
    "total_reviews_with_tag",
    "Recommended",
    "Not Recommended",
    "not_recommended_rate"
]])

print("\nTop 3 Positive Tags")
print(top_positive[[
    "tag",
    "total_reviews_with_tag",
    "Recommended",
    "Not Recommended",
    "recommended_rate"
]])
