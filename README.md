# 🎮Project: Steam Review Data Analysis

**Hypothesis** 

By fixing and controlling qualities a huge population of players dislike, the recommend rate of the game will significantly improve.

**Summary**

This project simulates a real-world data analysis workflow in the gaming industry. Using AI-powered large language model (LLM) tagging and statistical modeling, the project explores how user feedback relates to recommendation behavior, offering insight that can guide future game design decisions.

**Objectives**
- Extract real user reviews from Steam via web scraping  
- Clean and preprocess unstructured review text  
- Use a local LLM to auto-tag sentiment and themes  
- Explore probabilistic modeling (e.g., binomial regression) on user feedback  
- Identify pain points and improvement opportunities for game design

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

We collect data based on top-rated reviews for an initial signal on what delighted/annoyed players most. Starting with top-rated (“Most Helpful”) reviews maximizes the signal-to-noise ratio (SNR): high upvotes signal wider agreement, so these comments carry higher signal and product relevance in other words **these comments are broadly agreed, making them most actionable**

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

headers = {
    "User-Agent": (
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

<mark>⚠️Due  to legal and ethical reasons, the real Steam review data that had been extracted will not be included in this project. </br>
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

**Why this step**

Tagging unstructured review text into analyzable categories makes it possible for us to quantify recurring themes in player feedback. To do this at scale and with quick turnaround, we automate the tagging with an AI LLM since manual tagging is time-consuming and costly. Running this model locally preserves data privacy where no content is sent to third-party services.


