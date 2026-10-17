# 💼 Job Market Analysis and Recommendation System

> Analyzing Upwork job postings and recommending the most relevant jobs from a title or skill — with an interactive Streamlit app.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)

---

## 📌 Overview

This project explores the freelance job market on **Upwork** (postings from **7 Feb 2024 to 24 Mar 2024**) and turns the data into a **content-based job recommendation system**.

It has two parts:

1. **Job market analysis** (Jupyter Notebook) – cleaning, text preprocessing, and univariate / bivariate / multivariate analysis with written insights.
2. **Recommendation system + web app** (Streamlit) – type a job title or skill and instantly get the most similar jobs, with filters for job type and country.

## 🖼️ App Preview

![Job Recommender Streamlit App](<Job Recommandation System Streamlit.png>)

## ✨ Features

- **🔎 Search Jobs:** Easily find work by typing job titles or skills (like *Python developer*, *SEO*, or *WordPress*).
- **🎯 Best Matches First:** It uses a smart system to rank jobs so the most relevant ones show up at the top.
- **🎛️ Easy Filters:** Customize your search by choosing:
  - How many jobs to see (5 to 50)
  - Job type (*Hourly*, *Fixed-price*, or *All*)
  - Country
- **💰 Clear Pay & Location:** Every job card clearly shows the pay rate (hourly or fixed budget) and where the client is from.
- **🔗 Direct Links:** Click **"View job"** to go straight to the original Upwork job post.
- **⚡ Super Fast:** It processes the data once and saves it, so your future searches happen instantly.

## 🗂️ Dataset

| Property | Details |
|---|---|
| File | `all_upwork_jobs_2024-02-07-2024-03-24.csv` |
| Size | 244,828 rows × 8 columns |
| Period | 7 Feb 2024 – 24 Mar 2024 |


## 🛠️ Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, scikit-learn (TF-IDF, cosine similarity, RidgeCV, LDA)
- **NLP:** NLTK (stopwords, WordNet lemmatizer)
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **App:** Streamlit

## 📁 Project Structure

```
.
├── app.py                                          # Streamlit recommendation app
├── Job_Recommandation.ipynb                        # Analysis + model notebook
├── Job Market Analysis and Recommendation System.pptx  # Project presentation
├── Job Recommandation System Streamlit.png         # App screenshot
├── requirements.txt                                # App dependencies
├── LICENSE                                         # MIT License
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vm17-lab/Project_8_NHIS---Job-Market-Analysis-and-Recommendation-System.git
cd Project_8_NHIS---Job-Market-Analysis-and-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `all_upwork_jobs_2024-02-07-2024-03-24.csv` in the project root.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Explore the analysis

```bash
jupyter notebook Job_Recommandation.ipynb
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
