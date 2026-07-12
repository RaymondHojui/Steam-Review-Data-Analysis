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

<br>

> **Implementation**
> [`View the full Python script →`](src/scrape_clean.py)
> Scrapes Steam reviews, extracts review metadata, cleans the review text, and exports the raw and processed datasets.
> 
<br>

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

> **Implementation**
>
> [`View the full Python script →`](src/auto_tagging.py)
> Uses a locally hosted language model through Ollama to generate topic labels for each Steam review.

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

> **Implementation**
>
> [`View the full Python script →`](src/normalize.py)  
> Maps synonymous LLM-generated labels to a consistent set of canonical tags.

A limitation of this strategy is that, in extrme cases, some synonyms may be missed. For example, we might classify “user interface,” “UI,” “UI design,” “visual design” .etc under the tag "UI", but missed the tag “interface,” which should also classified as UI. That said, this is likely negligible because most synonyms are correctly captured, and only a small number are missed, so the overall results are largely unaffected.

**Problem 2:**
Although most labels generated are accurate, there may still be instances of inaccurate or low-quality labeling. This can occur because the LLM may have difficulty recognizing sarcasm or indirect comments. For example, a comment such as "I love how the enemies know where I am before I spawn" is referring to unfair detection, but the LLM might incorrectly label it as "Smart AI"

To invsetigate in the accuracy of LLM label, we can use simple random sampling to estimate a population proportion of correctness of LLM tagging. The way we do this is to randomly select 15% of the labled comments and mannually check if they are lablled correctly or incorrectly. Hence, we may use the following code to pick 15 random comments (since our sample size is 100) and check the correctness of AI lableing.

---
# 📊 Data Visualization
Now that we have the cleaned all the data, we have enough information to take a look at how often each tag appears in negative reviews. We can present this with a simple bar chart that display the frequency of each tag in `Not Recommed`.

> **Implementation**
>
> [`View the full Python script →`](src/visualization.py)  
> Counts recurring themes in negative reviews and visualizes their frequencies.

---
# 🔎 Figuring Key Qualities

We will perform a Binomial Logistic Regression for this project.

Why Binomial Logistic regression: Player recommendations on Steam are inherently binary: `Recommended` or `Not Recommended`. This makes **logistic regression** is a great modeling choice. (Logistic regression can estimate the probability that a review is `Not Recommended` based on the presence of specific issue tags)

Specifically, we will compare binomial `recommend` and `not-recommend` rates within each tag group, then separating top negative and top positive tags.

> **Implementation**
>
> [`View the full Python script →`](src/key_qualities.py)  
> Compares recommendation rates across review tags to identify potential positive and negative review drivers.

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

---
## ⚖️ License and Attribution

Copyright © 2026 Raymond Ho-Jui.

This project is licensed under the
[GNU Affero General Public License v3.0](LICENSE).

You may use, modify, and build upon this project under the terms of the
license. Redistributions must preserve the applicable copyright and license
notices, and modified versions must identify that changes were made.

If this project or part of its code contributes to your work, please credit:

> Built using code from **Steam Review Data Analysis** by **Raymond Ho-Jui**.
