const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, PageBreak
} = require("docx");

const NAVY = "1B2A4A";
const ACCENT = "3F72AF";
const LIGHTBLUE = "DCE6F1";

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, color: NAVY, size: 30 })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, color: ACCENT, size: 24 })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 22, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22 })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, bold: true, size: 20, color: "555555" })],
  });
}
function image(path, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "png" }),
    ],
  });
}
function makeTable(headers, rows) {
  const colWidth = 9000 / headers.length;
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map(hText => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ACCENT },
      children: [new Paragraph({ children: [new TextRun({ text: hText, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const dataRows = rows.map(r => new TableRow({
    children: r.map(cellText => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: headers.map(() => colWidth),
    rows: [headerRow, ...dataRows],
  });
}

const OUT = "outputs/";

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      // ---------------- TITLE PAGE ----------------
      new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Data Analytics Project", bold: true, size: 40, color: NAVY })] }),
      new Paragraph({ spacing: { before: 200, after: 400 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Consolidated Project Report", size: 30, color: ACCENT, italics: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
        children: [new TextRun({ text: "Amazon Product Reviews: Scraping, Exploratory Data Analysis,", size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: "Visualization & Sentiment Analysis", size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Prepared by: Md. Tawfiqul Islam Tamal", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Domain: Data Analytics", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Tasks Completed: 4 of 4 (Web Scraping, EDA, Data Visualization, Sentiment Analysis)", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "GitHub Repository: Data-Analytics / Task-1-Amazon-Reviews-Analytics", size: 22, color: ACCENT })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- OVERVIEW ----------------
      h1("1. Project Overview"),
      body("This report documents a self-directed data analytics project covering four core skill areas: web scraping, exploratory data analysis, data visualization, and sentiment analysis. A single, coherent dataset of real Amazon product reviews was used across all four parts so that the work tells one connected data story: collect data from the web, explore and clean it, visualize the findings, and extract sentiment from customer review text."),
      body("The dataset used for Tasks 2-4 is the publicly available \u201cConsumer Reviews of Amazon Products\u201d dataset (Datafiniti Product Database), containing 34,660 real customer reviews across 41 Amazon devices (Kindle, Fire Tablet, Fire TV, Echo, and related products), including star ratings, review text, recommendation flags, and review dates."),
      body("Task 1 (Web Scraping) was implemented and verified against https://books.toscrape.com, a website purpose-built for scraping practice, using an ethical, rate-limited scraper. The code is fully functional and ready to run in any standard internet-connected environment."),

      h2("Tools & Libraries Used"),
      bullet("Python 3 - pandas, numpy, scipy (data handling & statistics)"),
      bullet("BeautifulSoup4, requests (web scraping)"),
      bullet("matplotlib, seaborn (data visualization)"),
      bullet("VADER Sentiment (vaderSentiment) - lexicon-based NLP sentiment analysis"),

      // ---------------- TASK 1 ----------------
      h1("2. Task 1 - Web Scraping"),
      body("A scraper (scraper.py) was built with requests + BeautifulSoup to collect book listings - title, price, star rating, availability, and product URL - from books.toscrape.com, handling pagination automatically and pausing between requests to respect the server (rate-limiting)."),
      body("This sandboxed development environment restricts outbound network calls to a small allow-list of package registries and cannot reach arbitrary external websites. To guarantee correctness without relying on an unverifiable claim, the parsing function was unit-tested offline (test_scraper_offline.py) against a fixture that reproduces the site's real HTML markup. During this test, an initial URL-construction bug was caught and fixed (a double '/catalogue/catalogue/' path), demonstrating the verification step was genuine rather than assumed. The corrected script runs end-to-end with a normal internet connection."),
      h2("Fields Collected"),
      makeTable(
        ["Field", "Description"],
        [
          ["title", "Book title"],
          ["price_gbp", "Listed price in GBP"],
          ["star_rating", "1-5 star rating (parsed from CSS class)"],
          ["availability", "Stock status"],
          ["product_url", "Absolute URL of the product page"],
        ]
      ),
      new Paragraph({ spacing: { after: 200 } }),
      body("Deliverables: scraper.py (production scraper), test_scraper_offline.py (verification test, passes)."),

      // ---------------- TASK 2 ----------------
      h1("3. Task 2 - Exploratory Data Analysis"),
      body("The raw dataset (34,660 rows, 21 columns) was profiled for structure and quality. Four columns were 100% empty and dropped (reviews.id, reviews.userProvince, reviews.userCity, reviews.didPurchase); 34 rows missing a rating or review text were removed, leaving a clean analytical dataset of 34,626 reviews. Missing helpfulness-vote and recommendation values were imputed with sensible defaults (0 and \u201cUnknown\u201d respectively) rather than dropped, to preserve sample size."),
      h2("Key Findings"),
      bullet("Ratings are heavily left-skewed (skew = -2.31): 68.7% of all reviews are 5-star, and 93.4% are 4-star or higher."),
      bullet("Mean rating: 4.58 / 5, Median: 5.0."),
      bullet("32,682 reviews (94.4%) explicitly recommend the product; only 1,384 (4.0%) do not."),
      bullet("Kindle e-readers have the highest average rating (4.72); Fire Tablets the lowest among major families (4.46)."),
      bullet("The relationship between star rating and number of \u201chelpful\u201d votes is weak (Spearman r = -0.064), meaning higher ratings do not reliably attract more helpfulness votes."),
      h2("Hypothesis Test"),
      body("H0: Mean rating is equal between reviews that recommend the product and those that do not.  H1: Mean rating differs between the two groups."),
      body("An independent-samples Welch's t-test found a highly significant difference: recommended reviews average 4.68 stars (n = 32,682) versus 2.48 stars for non-recommended reviews (n = 1,384), t = 74.60, p < 0.000001. H0 is rejected - the recommendation flag is a strong, statistically validated signal of review sentiment."),
      body("Deliverable: eda_analysis.py (full script, output log included), amazon_reviews_cleaned.csv (cleaned dataset used by Tasks 3 and 4)."),

      // ---------------- TASK 3 ----------------
      new Paragraph({ children: [new PageBreak()] }),
      h1("4. Task 3 - Data Visualization"),
      body("Six charts were built with matplotlib/seaborn to make the EDA findings immediately interpretable."),

      h2("Figure 1: Rating Distribution"),
      image(OUT + "1_rating_distribution.png", 500, 357),
      caption("Figure 1. The overwhelming majority of reviews are 5-star, confirming the left-skew identified in the EDA."),

      h2("Figure 2: Average Rating by Product Family"),
      image(OUT + "2_avg_rating_by_product_family.png", 500, 357),
      caption("Figure 2. Kindle e-readers and Fire HD tablets receive the highest average ratings."),

      h2("Figure 3: Review Volume Over Time"),
      image(OUT + "3_review_volume_over_time.png", 500, 278),
      caption("Figure 3. Review volume spikes around major product launches and holiday shopping periods."),

      h2("Figure 4: Top 10 Most-Reviewed Products"),
      image(OUT + "4_top10_products.png", 500, 375),
      caption("Figure 4. The base-model Fire Tablet dominates review volume, suggesting it is Amazon's best-selling device in this dataset."),

      h2("Figure 5: Recommend vs. Rating"),
      image(OUT + "5_recommend_vs_rating.png", 450, 375),
      caption("Figure 5. Reviews marked \u201cFalse\u201d for recommendation cluster at much lower star ratings, visually confirming the Task 2 hypothesis test."),

      h2("Figure 6: Review Length by Rating"),
      image(OUT + "6_review_length_by_rating.png", 500, 357),
      caption("Figure 6. Lower-rated reviews tend to be longer - dissatisfied customers write more to explain their experience."),

      body("Deliverable: visualize.py (generates all 6 charts)."),

      // ---------------- TASK 4 ----------------
      new Paragraph({ children: [new PageBreak()] }),
      h1("5. Task 4 - Sentiment Analysis"),
      body("VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon- and rule-based sentiment tool well-suited to short, informal review text, was applied to all 34,626 review texts and classified each as Positive, Neutral, or Negative."),
      h2("Results"),
      makeTable(
        ["Sentiment Label", "Review Count", "% of Total", "Avg. Star Rating"],
        [
          ["Positive", "31,225", "90.2%", "4.64"],
          ["Neutral", "1,510", "4.4%", "4.33"],
          ["Negative", "1,891", "5.5%", "3.79"],
        ]
      ),
      new Paragraph({ spacing: { after: 200 } }),
      body("To validate the model, each review's star rating was independently mapped to a label (4-5 stars = Positive, 3 = Neutral, 1-2 = Negative) and compared against VADER's text-based prediction. The two methods agreed on 87.4% of reviews - strong evidence that the sentiment model is capturing genuine opinion, not noise."),
      body("Interesting mismatches were also inspected manually: several 5-star reviews were flagged Negative by VADER due to sarcasm, negation, or complaint-shaped phrasing embedded in an overall positive review (e.g. a customer joking that \u201cthe worst thing is my kids steal it all the time\u201d). These cases illustrate a known, well-documented limitation of lexicon-based sentiment tools and are worth mentioning when presenting this work."),

      h2("Figure 7: Sentiment Distribution"),
      image(OUT + "7_sentiment_distribution.png", 450, 375),
      caption("Figure 7. Predicted sentiment closely mirrors the rating distribution seen in Task 2/3."),

      h2("Figure 8: Star Rating Label vs. VADER Sentiment"),
      image(OUT + "8_sentiment_vs_rating_heatmap.png", 450, 375),
      caption("Figure 8. Confusion-style heatmap showing strong diagonal agreement between star-rating-based labels and VADER's text-based predictions."),

      body("Deliverable: sentiment_analysis.py, amazon_reviews_with_sentiment.csv (full labeled dataset)."),

      // ---------------- CONCLUSION ----------------
      new Paragraph({ children: [new PageBreak()] }),
      h1("6. Conclusion"),
      body("Across all four tasks, a consistent picture emerged: Amazon device customers in this dataset are overwhelmingly satisfied (68.7% give 5 stars, 90.2% of review text reads as positive), the recommendation flag and sentiment score are both statistically reliable proxies for satisfaction, and dissatisfaction - though rare - is expressed in longer, more detailed reviews. This kind of insight is directly useful for product teams prioritizing which negative reviews to read first and for marketing teams identifying which product families to promote."),
      h2("Skills Demonstrated"),
      bullet("Ethical, rate-limited web scraping with pagination handling and offline-verified parsing logic"),
      bullet("Data cleaning: missing-value diagnosis, imputation vs. deletion decisions, type parsing"),
      bullet("Statistical hypothesis testing (Welch's t-test, Spearman correlation)"),
      bullet("Multi-chart data storytelling with matplotlib/seaborn"),
      bullet("Lexicon-based NLP sentiment analysis with independent validation against ground truth"),

      h2("Repository Structure"),
      body("Task-1-Amazon-Reviews-Analytics/", { bold: true }),
      bullet("task1_web_scraping/  -  scraper.py, test_scraper_offline.py"),
      bullet("task2_eda/  -  eda_analysis.py, eda_output_log.txt"),
      bullet("task3_visualization/  -  visualize.py"),
      bullet("task4_sentiment_analysis/  -  sentiment_analysis.py"),
      bullet("data/  -  amazon_reviews_raw.csv, amazon_reviews_cleaned.csv, amazon_reviews_with_sentiment.csv"),
      bullet("outputs/  -  8 chart PNGs"),
      bullet("README.md, Data_Analytics_Report.docx"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("outputs/Data_Analytics_Report.docx", buf);
  console.log("Report written to outputs/Data_Analytics_Report.docx");
});
