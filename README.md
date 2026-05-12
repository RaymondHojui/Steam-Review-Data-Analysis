# Identifying Positive and Negative Review Drivers from LLM-Tagged Steam Reviews
**Project Goal** 

Use player feedback data to identify the game qualities most associated with positive and negative recommendations by comparing recommend rates across LLM-generated review tags.

**Summary**

This project simulates a real-world data analysis workflow in the gaming industry. Using AI-powered large language model (LLM) tagging and binomial rate analysis, the project explores how user feedback themes relate to Steam recommendation behavior. The analysis identifies tags with the highest `Recommended` rates and tags with the highest `Not Recommended` rates, offering insight into possible pain points and strengths in the game experience.

**Objectives**

- Extract real user reviews from Steam via web scraping  
- Clean and preprocess unstructured review text  
- Use a local LLM to auto-tag sentiment and themes  
- Use binomial rate analysis to compare `Recommended` and `Not Recommended` outcomes across review tags 
- Identify review themes most associated with negative and positive recommendation outcomes

**Software and Tools:**

- **Python** (Pandas, BeautifulSoup, Counter, Matplotlib)
- **LLM** (DeepSeek) AI used for auto-tagging unstructured text
- **CSV** for data storage and versioning

**Database Schema**

- **Raw Data**  
  Source: Scraped directly from Steam reviews.  
  Fields: `user_name`, `recommend`, `hours`, `date`, `review`  
  Purpose: Preserve the original structure and metadata for reference.

- **Cleaned Data**  
  Source: Raw Data after text preprocessing.  
  Changes: Removed date prefixes in review text and standardized formats.  
  Fields: `user_name`, `recommend`, `hours`, `date`, `review`  
  Purpose: Prepare consistent input for LLM tagging.

- **Final Data**  
  Source: Cleaned Data with additional sentiment/theme labels.  
  Fields: `user_name`, `recommend`, `hours`, `date`, `review`, `llm_labels`  
  Purpose: Purpose: This is the final dataset used for statistical modeling. It contains both the cleaned reviews and AI-generated sentiment/theme labels (via a large language model, LLM).

---
## 🕷️Steam Review Scraper & Cleaning Pipeline

First, we scrape top-rated Steam user reviews using BeautifulSoup and save the raw data as `raw_reviews.csv`. Then, we clean the review text using a regular expression to remove date prefixes and save the result as `reviews_cleaned.csv`.

**Why this step**

We collect a reproducible sample of Steam reviews and convert the text and metadata (playtime, recommendation, etc.) into structured features for future tagging and modeling.

**Why start with “Top-Rated / Most Helpful”**

We collect data based on top-rated reviews for an initial signal on what delighted/annoyed players most. Starting with top-rated (“Most Helpful”) reviews maximizes the signal-to-noise ratio (SNR): high upvotes signal wider agreement, so these comments carry higher signal and product relevance. In other words **these comments are broadly agreed, making them most actionable**

**Code:**

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Part 1: Web Scraping

# Replace with a valid Steam App ID
APP_ID = "<APP_ID>"  # <-- Replace this before running
url = f"https://steamcommunity.com/app/{APP_ID}/reviews/?browsefilter=toprated"

headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}

# Send GET request to the review page
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Extract reviews from HTML
review_cards = soup.find_all("div", class_="apphub_UserReviewCardContent")
reviews = []

for card in review_cards:
    title_div = card.find_previous("div", class_="title")
    recommend = title_div.get_text(strip=True) if title_div else "Unknown"

    hours_div = card.find_previous("div", class_="hours")
    hours_played = hours_div.get_text(strip=True) if hours_div else "Unknown"

    content_div = card.find("div", class_="apphub_CardTextContent")
    content = content_div.get_text(separator=" ", strip=True) if content_div else "No Content"

    date_div = card.find("div", class_="date_posted")
    date_posted = date_div.get_text(strip=True) if date_div else "Unknown"

    author_div = card.find_next("div", class_="apphub_CardContentAuthorName")
    user_name = author_div.get_text(strip=True) if author_div else "Anonymous"

    reviews.append({
        "user_name": user_name,
        "recommend": recommend,
        "hours": hours_played,
        "date": date_posted,
        "review": content
    })

# Convert to DataFrame
df_raw = pd.DataFrame(reviews)

# Save raw data
raw_file_path = "raw_reviews.csv"
try:
    with open(raw_file_path, 'x', encoding="utf-8-sig") as f:
        df_raw.to_csv(f, index=False)
    print(f"First-time write: saved to {raw_file_path}")
except FileExistsError:
    df_raw.to_csv(raw_file_path, mode='a', header=False, index=False)
    print(f"File exists: data appended to {raw_file_path}")

print("Raw Data Preview:")
print(df_raw.head())

#Part 2: Cleaning Review Text

df_cleaned = df_raw.copy()

# Remove "Posted: Month Day" from the start of the review (e.g., "Posted: March 22")
df_cleaned["review"] = df_cleaned["review"].str.replace(
    r"^Posted:\s*[A-Za-z]+\s+\d{1,2}\s*", 
    "", 
    regex=True
)

# Save cleaned data
cleaned_file_path = "reviews_cleaned.csv"
df_cleaned.to_csv(cleaned_file_path, index=False, encoding="utf-8-sig")
print(f"Cleaned data saved to {cleaned_file_path}")

print("Cleaned Review Preview:")
print(df_cleaned["review"].head(5))


```
The **raw data** set is saved under `raw_reviews.csv` and the **cleaned data** set is saved under `reviews_cleaned.csv`

<mark>⚠️Due  to legal and ethical reasons, the real Steam review data that I have extracted will not be included in this project. </br>
The following is an example illustrating the structure and format of the `reviews_cleaned.csv` dataset used in this analysis for demonstration purposes; it does not contain real user data. </mark>


| user_name    | recommend       | hours               | date              | review                                                                 |
|--------------|------------------|----------------------|-------------------|--------------------------------------------------------------------------------|
| DragonSlayer | Recommended      | 102.5 hrs on record  | Posted: April 10  | One of the best co-op experiences I've had in years...                         |
| CoffeeAddict | Not Recommended  | 5.2 hrs on record    | Posted: March 3   | Game crashes every 10 minutes on my laptop...                                  |
| PixelWizard  | Recommended      | 210.0 hrs on record  | Posted: May 17    | A true hidden gem. The pixel art is beautiful...                               |
| AFK_Ninja    | Mixed            | 47.3 hrs on record   | Posted: June 1    | Great mechanics, but the matchmaking is trash...                               |
| GlitchHunter | Not Recommended  | 13.7 hrs on record   | Posted: May 29    | This game has potential, but it’s buried under bugs and UI issues...          |

(see full data frame in `reviews_cleaned.csv`)

Now we have cleaned sorted data that is ready to be tagged by an llm!

---
## 🤖 Auto tagging with Local LLM
Next, we will run a local LLM over the `review` field (column) for each record (row) and assign 1–5 specific labels summarizing the content. Labels are stored as a list in a new CSV field named`llm_lable`

**Preparation and Getting Ready**
This project runs the `deepseek-r1:14b` LLM locally using the open-source tool `ollama`. While we could use other models or host the LLM via external APIs, we chose `deepseek-r1:14b` because it is open-source, and we run it locally to preserve data privacy (no content ran locally is sent to third-party services). Specific steps to find how to use ollama can be accessed here [].

**Why this step**

Tagging unstructured review text into analyzable categories makes it possible for us to quantify recurring themes in player feedback. To do this at scale and with quick turnaround, we automate the tagging with an AI LLM since manual tagging is time consuming and costly in real world senarios. Running this model locally preserves data privacy where no content is sent to third-party services.

**Code:**
```python
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
```
After running the code, a new csv file called `reviews_with_llm_labels.csv` should be made, containg all the tags created by the LLM.
It should look something like this:

| user_name    | recommend       | hours               | date             | review                                                               | llm_labels                              |
| ------------ | --------------- | ------------------- | ---------------- | -------------------------------------------------------------------- | --------------------------------------- |
| DragonSlayer | Recommended     | 102.5 hrs on record | Posted: April 10 | One of the best UI I've seen in years...                             | ["user interface", "positive"]          |
| CoffeeAddict | Not Recommended | 5.2 hrs on record   | Posted: March 3  | Game crashes every 10 minutes on my laptop...                        | ["crash", "stability", "performance"]   |
| PixelWizard  | Recommended     | 210.0 hrs on record | Posted: May 17   | A true hidden gem. The pixel art is beautiful...                     | ["art", "aesthetics", "positive", "pixle art"]       |
| AFK_Ninja    | Recommend       | 47.3 hrs on record  | Posted: June 1   | Great mechanics, but the matchmaking is trash...                     | ["matchmaking", "multiplayer", "mechanics"] |
| GlitchHunter | Not Recommended | 13.7 hrs on record  | Posted: May 29   | This game has potential, but it’s buried under bugs and UI issues... | ["bug", "ui", "performance"]            |


(see full data frame in `reviews_cleaned.csv`)

**Issues About LLM lables & How to Fix It**

There are 2 major issues with the leabels at this stage:

**Problem 1:**
Different labels are being generated for the same concept (e.g., “user interface” and “UI”). This occurs because the LLM reviews are stateless and don’t remember previous tag decisions. However, there is a recognizable pattern for the tags. Tags are generally 1-2 word long, and similar words will keep repeating. A simple fix is to normalize synonyms to a single canonical tag.

```python
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
```
A limitation of this strategy is that, in extrme cases, some synonyms may be missed. For example, we might classify “user interface,” “UI,” “UI design,” “visual design” .etc under the tag "UI", but missed the tag “interface,” which should also classified as UI. That said, this is likely negligible because most synonyms are correctly captured, and only a small number are missed, so the overall results are largely unaffected.

**Problem 2:**
Although most labels generated are accurate, there may still be instances of inaccurate or low-quality labeling. This can occur because the LLM may have difficulty recognizing sarcasm or indirect comments. For example, a comment such as "I love how the enemies know where I am before I spawn" is referring to unfair detection, but the LLM might incorrectly label it as "Smart AI"

To invsetigate in the accuracy of LLM label, we can use simple random sampling to estimate a population proportion of correctness of LLM tagging. The way we do this is to randomly select 15% of the labled comments and mannually check if they are lablled correctly or incorrectly. Hence, we may use the following code to pick 15 random comments (since our sample size is 100) and check the correctness of AI lableing.

---
# 📊 Data Visualization
Now that we have the cleaned all the data, we have enough information to take a look at how often each tag appears in negative reviews. We can present this with a simple bar chart that display the frequency of each tag in `Not Recommed`.

**Code:**
```python
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

```
---
# 🔎 Figuring Key Qualities

We will perform a Binomial Logistic Regression for this project.

Why Binomial Logistic regression: Player recommendations on Steam are inherently binary: `Recommended` or `Not Recommended`. This makes **logistic regression** is a great modeling choice. (Logistic regression can estimate the probability that a review is `Not Recommended` based on the presence of specific issue tags)

Specifically, we will compare binomial `recommend` and `not-recommend` rates within each tag group, then separating top negative and top positive tags.

**Code:**
```python
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
```
After running the code, it will output two ranked tables:

1. Top 3 Negative Tags — tags with the highest `Not Recommended` rate
2. Top 3 Positive Tags — tags with the highest `Recommended` rate

Each row represents a tag, not an individual review. For each tag, the code counts how many reviews containing that tag were `Recommended` or `Not Recommended`, then calculates the corresponding rate.

### Top 3 Negative Tags

| tag | total_reviews_with_tag | Recommended | Not Recommended | not_recommended_rate |
| --- | ---------------------: | ----------: | --------------: | -------------------: |
| complaints | 12 | 1 | 11 | 91.7% |
| suggestions | 7 | 1 | 6 | 85.7% |
| monetization | 20 | 4 | 16 | 80.0% |

For these tags, we treat `Not Recommended` as the binomial “success.”

Example: `X_complaints ~ Binomial(n = 12, p = 0.917)`

This means that among reviews tagged with `complaints`, 11 out of 12 were `Not Recommended`.

### Top 3 Positive Tags

| tag | total_reviews_with_tag | Recommended | Not Recommended | recommended_rate |
| --- | ---------------------: | ----------: | --------------: | ---------------: |
| difficulty | 11 | 8 | 3 | 72.7% |
| balance | 5 | 3 | 2 | 60.0% |
| online | 6 | 3 | 3 | 50.0% |

For these tags, we treat `Recommended` as the binomial “success.”

Example: `X_difficulty ~ Binomial(n = 11, p = 0.727)`

This means that among reviews tagged with `difficulty`, 8 out of 11 were `Recommended`.
