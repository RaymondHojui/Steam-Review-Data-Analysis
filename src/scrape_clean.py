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

