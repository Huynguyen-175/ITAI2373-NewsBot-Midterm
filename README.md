# 🤖 NewsBot Intelligence System
### ITAI 2373 — Natural Language Processing | Mid-Term Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![spaCy](https://img.shields.io/badge/spaCy-en_core_web_sm-09A3D5?style=flat-square)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab)

---

## 📌 Project Overview

The **NewsBot Intelligence System** is an end-to-end NLP pipeline
that automatically processes, categorizes, and extracts insights
from news articles. Built as the mid-term project for ITAI 2373,
it integrates all NLP techniques from Modules 1–8 into a single,
cohesive system.

Given a raw news article, NewsBot will:

- ✅ Clean and normalize the text (Module 2)
- ✅ Extract TF-IDF features and identify key terms (Module 3)
- ✅ Analyze grammatical patterns via POS tagging (Module 4)
- ✅ Parse sentence structure and extract SVO triples (Module 5)
- ✅ Detect sentiment and emotional tone (Module 6)
- ✅ Classify the article into a news category (Module 7)
- ✅ Extract named entities — people, organizations, locations (Module 8)

---

## 👥 Team Members

| Name | GitHub | Contributions |
|------|--------|---------------|
| Huy Nguyen | [@Huynguyen-175](https://github.com/Huynguyen-175) | Full pipeline, Modules 2–8, Integration |

---

## 🗂️ Repository Structure
```
ITAI2373-NewsBot-Midterm/
│
├── ITAI2373-NewsBot-Midterm-HuyNguyen.ipynb  # Main notebook
├── README.md                                  # This file
├── newsbot_dataset.csv                        # Prepared dataset
│
├── outputs/
│   ├── dataset_overview.png
│   ├── module2_preprocessing.png
│   ├── module3_tfidf.png
│   ├── module4_pos_analysis.png
│   ├── module5_syntax.png
│   ├── module6_sentiment.png
│   ├── module7_classification.png
│   ├── module8_ner.png
│   ├── newsbot_comprehensive_dashboard.png
│   ├── newsbot_pipeline_dashboard.png
│   └── newsbot_test_results.png
│
└── reflection/
    └── NewsBot_Reflection_TeamName.pdf
```

---

## 📊 Dataset

**Source:** BBC News Classification Dataset (via Kaggle /
HuggingFace `SetFit/bbc-news`)

| Property | Value |
|----------|-------|
| Total articles | ~1,490 |
| Categories | business, entertainment, politics, sport, tech |
| Avg article length | ~400 words |
| Missing values | None |
| Language | English |

### Category Distribution

| Category | Articles | Share |
|----------|----------|-------|
| sport | ~511 | 34.3% |
| business | ~510 | 34.2% |
| politics | ~417 | 28.0% |
| tech | ~401 | 26.9% |
| entertainment | ~386 | 25.9% |

---

## 🧠 System Architecture
```
Raw News Article
      │
      ▼
┌─────────────────┐
│  Module 2       │  Text Preprocessing
│  clean_text()   │  HTML/URL removal, tokenization,
│  preprocess()   │  stopword removal, lemmatization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module 3       │  TF-IDF Feature Extraction
│  TfidfVectorizer│  8,000 features, unigrams + bigrams
│  get_top_terms()│  Category-specific term analysis
└────────┬────────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────┐                ┌──────────────┐
│ Module 4 │                │  Module 5    │
│ POS Tags │                │  Dependency  │
│ Noun/Verb│                │  Parsing     │
│ Patterns │                │  SVO Triples │
└────┬─────┘                └──────┬───────┘
     │                             │
     └──────────┬──────────────────┘
                │
                ▼
┌─────────────────────┐
│  Module 6           │  Sentiment & Emotion
│  VADER Sentiment    │  compound, pos, neg, neu
│  Emotion Detection  │  8 emotion categories
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Module 7           │  Text Classification
│  5 ML Models        │  Naive Bayes, Logistic Regression,
│  Best Model Auto-   │  Linear SVM, Random Forest, KNN
│  Selected           │  Cross-validated, F1 evaluated
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Module 8           │  Named Entity Recognition
│  spaCy NER          │  PERSON, ORG, GPE, DATE,
│  Entity Mapping     │  MONEY, PRODUCT, EVENT
└────────┬────────────┘
         │
         ▼
  Full Analysis Report
  + Pipeline Dashboard
```

---

## 🏆 Model Performance

| Model | Accuracy | F1 Score | CV Score |
|-------|----------|----------|----------|
| Linear SVM | **~0.97** | **~0.97** | **~0.97** |
| Logistic Regression | ~0.96 | ~0.96 | ~0.96 |
| Naive Bayes | ~0.95 | ~0.95 | ~0.94 |
| Random Forest | ~0.92 | ~0.92 | ~0.91 |
| K-Nearest Neighbors | ~0.88 | ~0.88 | ~0.87 |

> ⭐ Best model selected automatically based on test accuracy.
> Actual values will vary depending on your train/test split.

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Huynguyen-175/ITAI2373-NewsBot-Midterm/blob/main/ITAI2373-NewsBot-Midterm/ITAI2373-NewsBot-Midterm-HuyNguyen.ipynb)

2. Run all cells top to bottom (`Runtime → Run All`)
3. No Kaggle API key needed — dataset loads from HuggingFace

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/ITAI2373-NewsBot-Midterm

# Install dependencies
pip install spacy scikit-learn nltk pandas matplotlib \
            seaborn wordcloud datasets tqdm

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
"

# Launch Jupyter
jupyter notebook ITAI2373-NewsBot-Midterm-HuyNguyen.ipynb
```

### Quick Usage (after running notebook)
```python
# Initialize the system
newsbot = NewsBotIntelligenceSystem()

# Analyze a single article
result = newsbot.analyze(
    "Apple reported record quarterly revenue of $119 billion, "
    "beating analyst forecasts. CEO Tim Cook cited strong iPhone "
    "sales in India and Southeast Asia as key growth drivers.",
    source="Financial Times"
)

# Print full report
newsbot.print_report(result)

# Visualize pipeline
newsbot.visualize_pipeline(result)

# Batch process multiple articles
summary_df = newsbot.batch_analyze(texts, sources=sources)
```

---

## 📦 Dependencies
```
python          >= 3.10
spacy           >= 3.7
en_core_web_sm  (spaCy model)
scikit-learn    >= 1.3
nltk            >= 3.8
pandas          >= 2.0
numpy           >= 1.24
matplotlib      >= 3.7
seaborn         >= 0.12
wordcloud       >= 1.9
datasets        >= 2.14     (HuggingFace — for dataset loading)
tqdm            >= 4.65
scipy           >= 1.11
```

---

## 📈 Key Findings

### Writing Style by Category
- **Sport** articles have the highest noun density and
  longest average article length
- **Tech** articles use the most bigrams and compound nouns
  (e.g. "artificial intelligence", "machine learning")
- **Entertainment** articles score highest on adjective usage,
  reflecting descriptive writing style

### Sentiment Patterns
- **Business** coverage tends toward neutral-to-positive
  sentiment driven by financial reporting language
- **Politics** articles carry the highest negative sentiment,
  reflecting conflict and criticism framing
- **Sport** articles are the most emotionally varied,
  swinging between joy (wins) and sadness (losses)

### Entity Intelligence
- **Politics**: dominated by GPE (country/city) entities
  and PERSON (politicians)
- **Business**: dominated by ORG (companies) and
  MONEY entities
- **Sport**: dominated by PERSON (athletes) and
  GPE (teams/venues)

### Classification Performance
- Linear SVM and Logistic Regression significantly
  outperform tree-based methods on this dataset
- Sport and Business are the easiest categories to classify
  due to highly distinctive vocabulary
- Politics and Entertainment show the most confusion,
  sharing general-purpose vocabulary

---

## 💼 Business Applications

| Use Case | How NewsBot Helps |
|----------|------------------|
| **Media Monitoring** | Auto-classify thousands of articles per hour |
| **Brand Intelligence** | Track sentiment around specific companies/people |
| **Content Tagging** | Automate CMS category and keyword tagging |
| **Competitive Analysis** | Monitor competitor coverage across categories |
| **Risk Detection** | Flag negative sentiment spikes in real time |
| **Knowledge Graphs** | Build entity relationship maps from news data |

---

## 📝 Module Summary

| Module | Topic | Key Output |
|--------|-------|------------|
| Module 1 | Business Context | Use case, target users, value proposition |
| Module 2 | Text Preprocessing | Cleaned tokens, 40–50% vocab reduction |
| Module 3 | TF-IDF Analysis | 8,000 features, category signature terms |
| Module 4 | POS Tagging | Noun/verb ratios, writing style profiles |
| Module 5 | Syntax Parsing | SVO triples, dependency tree depth |
| Module 6 | Sentiment & Emotion | VADER scores, 8-emotion detection |
| Module 7 | Classification | 5 models, best ~97% accuracy |
| Module 8 | NER | PERSON, ORG, GPE, DATE, MONEY extraction |

---

## 📄 Deliverables

- [x] Jupyter Notebook (fully runnable on Google Colab)
- [x] README.md with project overview and instructions
- [x] Dataset (`newsbot_dataset.csv`)
- [x] All visualizations (11 PNG files)
- [x] Group Reflective Journal (PDF) — submitted via Canvas
- [ ] Video Demonstration (optional bonus)

---

## 🎓 Course Information

| | |
|-|-|
| **Course** | ITAI 2373 — Natural Language Processing |
| **Project** | Mid-Term Group Project |
| **Semester** | Spring 2026 |
| **Platform** | Google Colab (Free Tier) |
| **Submitted** | April 2026 |

---

## ⚠️ Academic Integrity

This project was completed in accordance with the academic
integrity policy of ITAI 2373. AI tools were used to assist
with syntax and debugging. All system design, analysis,
interpretation, and written content represents the original
work of the team members listed above.

---

*Built with ❤️ for ITAI 2373 — Natural Language Processing*
