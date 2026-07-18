# Task 1 — Amazon Product Reviews: Data Analytics Project

A personal end-to-end data analytics project covering **Web Scraping, Exploratory Data Analysis, Data Visualization, and Sentiment Analysis**, built around one connected dataset so the work tells a single data story: **collect data from the web → explore & clean it → visualize the findings → understand customer sentiment from review text.**

## Project Story

- **EDA, Visualization, and Sentiment Analysis** use the public **"Consumer Reviews of Amazon Products"** dataset (Datafiniti Product Database) — 34,660 real customer reviews of Amazon devices (Kindle, Fire Tablet, Fire TV, Echo).
- **Web Scraping** is a standalone scraper built and verified against [books.toscrape.com](https://books.toscrape.com), a site purpose-built for scraping practice.

Full write-up with charts, methodology, and findings: **`outputs/Data_Analytics_Report.docx`**

## Repository Structure

```
Task-1-Amazon-Reviews-Analytics/
├── task1_web_scraping/
│   ├── scraper.py                  # BeautifulSoup + requests scraper (books.toscrape.com)
│   └── test_scraper_offline.py     # Offline unit test proving the parser is correct
├── task2_eda/
│   ├── eda_analysis.py             # Full EDA: cleaning, stats, hypothesis testing
│   └── eda_output_log.txt          # Captured console output of the analysis
├── task3_visualization/
│   └── visualize.py                # Generates all 6 charts
├── task4_sentiment_analysis/
│   └── sentiment_analysis.py       # VADER sentiment classification + validation
├── data/
│   ├── amazon_reviews_raw.csv            # Original dataset
│   ├── amazon_reviews_cleaned.csv        # Output of EDA step, used downstream
│   └── amazon_reviews_with_sentiment.csv # Output of sentiment analysis step
├── outputs/
│   ├── 1_rating_distribution.png ... 8_sentiment_vs_rating_heatmap.png
│   └── Data_Analytics_Report.docx
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# Web Scraping — requires normal internet access
cd task1_web_scraping && python test_scraper_offline.py   # verify parser first
python scraper.py

# EDA
cd ../task2_eda && python eda_analysis.py

# Visualization (run after EDA, needs the cleaned CSV)
cd ../task3_visualization && python visualize.py

# Sentiment Analysis (run after EDA)
cd ../task4_sentiment_analysis && python sentiment_analysis.py
```

## Key Findings (see full report for details)

- 68.7% of all reviews are 5-star; mean rating is 4.58/5 (heavily left-skewed).
- Reviews that recommend the product average 4.68★ vs. 2.48★ for those that don't (Welch's t-test, p < 0.000001).
- VADER text-based sentiment agrees with star-rating-based sentiment on **87.4%** of reviews.
- Kindle e-readers have the highest average rating (4.72) among product families.

## Tech Stack

Python 3 · pandas · numpy · scipy · BeautifulSoup4 · requests · matplotlib · seaborn · vaderSentiment

---
**Author:** Md. Tawfiqul Islam Tamal
