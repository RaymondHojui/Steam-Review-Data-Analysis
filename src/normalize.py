import pandas as pd
import ast

csv_path = r"data/utf_final_result.csv"
df = pd.read_csv(csv_path)

# Normalize the tags here
label_mapping = {
    "multiplayer experience": "multiplayer",
    "multiplayer mode": "multiplayer",
    "co-op": "multiplayer",
    "coop": "multiplayer",
    "bugs": "bug",
    "optimization": "performance",
    "crash": "bug",
    "crashes": "bug",
    "fps": "performance",
    "frame rate": "performance",
    "online": "multiplayer",
    "price": "monetization",
    "cost": "monetization",
    "disapointment": "complaints",
    "challenge": "difficulty",
    "negativity": "complains",
    "visual": "graphics",
    "visuals": "graphics",
    "user interface": "ui",
    "balance": "difficulty",
}

def normalize_tag(tag):
    tag = tag.lower().strip()
    return label_mapping.get(tag, tag)

all_tags = []
for cell in df["llm_labels"]:
    if pd.isna(cell):
        continue
    try:
        tag_list = ast.literal_eval(cell)
    except:
        continue
    if isinstance(tag_list, str):
        tag_list = [tag_list]
    for t in tag_list:
        all_tags.append(normalize_tag(t))
