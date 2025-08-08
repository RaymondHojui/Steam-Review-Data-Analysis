# 🎮Projrct: Steam Review Data Analysis
**Summary**

Software and Tools:

- **Python** (Pandas, BeautifulSoup, Counter, Matplotlib)

- **LLM** (DeepSeek) for auto-tagging unstructured text

- **CSV** for data storage and versioning

In this project I have:

- Collected real user reviews from Steam using web scraping **and** cleaned/preprocessed the review data

- Used a local LLM (DeepSeek) to auto-generate multi-label tags for each review

- Built conceptual foundations for logistic regression and binomial modeling based on user sentiment distribution

- Interpreted the limitations of using probabilistic distributions (e.g., binomial) for real-world game reviews

---
## 🕷️Extraction of Steam User Reviews Through Scraping

First, we extract the data set using Python's BeautifulSoup libary on Steams Website.

```python


