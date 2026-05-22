# 🚀 Career Recommendation System

An intelligent and explainable career recommendation system built with Python and Streamlit.

This project analyzes a candidate’s:
- technical skills
- soft skills
- preferred career roles
- experience
- education
- English language level
- Danish language level
- preferred locations
- preferred employment types
- preferred work modes

and recommends the best matching job opportunities using a hybrid recommendation approach.

---

# 📌 Project Goal

The main goal of this project is to help candidates discover career opportunities that best match their profile through a hybrid recommendation system.

The system:
- analyzes the candidate profile
- evaluates technical and soft skill compatibility
- performs job match analysis
- identifies missing skills and skill gaps
- generates a personalized success roadmap
- provides interactive visual analytics and recommendation insights

to help users better understand:
- their strengths
- improvement areas
- career readiness
- learning priorities
- best matching career opportunities

The system combines:
- semantic similarity
- exact technical skill matching
- exact soft skill matching
- structured scoring logic

to generate explainable and personalized career recommendations.

---

# 🧠 Recommendation System Architecture

```text
USER PROFILE
      ↓

┌──────────────────────────────────────┐
│ Candidate Information                │
│                                      │
│ - Technical Skills                   │
│ - Soft Skills                        │
│ - Preferred Career Roles             │
│ - Experience                         │
│ - Education                          │
│ - English Level                      │
│ - Danish Level                       │
│ - Preferred Locations                │
│ - Preferred Work Modes               │
│ - Preferred Employment Types         │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Build User Profile                   │
│                                      │
│ technical_skills +                   │
│ soft_skills +                        │
│ preferred_roles                      │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Build Job Profiles                   │
│                                      │
│ technical_skills_required +          │
│ soft_skills_required +               │
│ role_category                        │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ TF-IDF Vectorization                 │
│                                      │
│ Convert text profiles into vectors   │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Cosine Similarity                    │
│                                      │
│ Measures semantic similarity         │
│ between user profile and jobs        │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Exact Skill Matching                 │
│                                      │
│ - Technical Skill Overlap            │
│ - Soft Skill Overlap                 │
│ - Missing Skills Detection           │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Structured Matching                  │
│                                      │
│ - Experience Match                   │
│ - Education Match                    │
│ - English Match                      │
│ - Danish Match                       │
│ - Location Match                     │
│ - Work Mode Match                    │
│ - Employment Type Match              │
└──────────────────────────────────────┘
      ↓

┌──────────────────────────────────────┐
│ Hybrid Final Scoring Engine          │
│                                      │
│ Combines all scores into             │
│ a final recommendation score         │
└──────────────────────────────────────┘
      ↓

FINAL CAREER RECOMMENDATIONS
```

---

# ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

---

# 🔍 Core Recommendation Logic

The system uses a hybrid recommendation approach.

## 1. Semantic Similarity

Uses:
- TF-IDF Vectorization
- Cosine Similarity

to measure semantic similarity between:
- user profile
- job profile

This helps the system understand profile relevance and contextual similarity.

---

## 2. Exact Skill Matching

Uses set intersection to calculate:
- matched technical skills
- missing technical skills
- matched soft skills
- missing soft skills

This layer improves:
- explainability
- skill gap analysis
- recommendation transparency

---

## 3. Structured Matching

Calculates additional scores for:
- experience level
- education level
- English language level
- Danish language level
- preferred locations
- preferred work modes
- preferred employment types

This layer applies business logic and user preferences.

---

# 📊 Final Score Formula

```python
final_score = (

    similarity_score * 0.35 +

    technical_score * 0.20 +

    soft_score * 0.10 +

    exp_score * 0.10 +

    education_score * 0.08 +

    english_score * 0.05 +

    danish_score * 0.05 +

    location_score * 0.03 +

    work_mode_score * 0.02 +

    employment_score * 0.02
)
```

---

# ✨ Features

- Hybrid recommendation engine
- Explainable AI recommendations
- Semantic job matching
- Exact skill matching
- Skill gap analysis
- Personalized success roadmap
- Interactive visual analytics
- Dynamic recommendation generation
- Career readiness scoring

---

# 📈 Visual Analytics

The system includes:
- Radar Charts
- Gauge Charts
- Donut Charts
- Match Category Analysis
- Missing Skills Visualization

These visualizations help users better understand:
- their strengths
- weaknesses
- career readiness
- improvement priorities

---

# 📂 Project Structure

```text
project/

│
├── app.py

│
├── data/

│   ├── cleaned_jobs_dataset.csv

│   ├── technical_skills.csv

│   └── soft_skills.csv

│
├── README.md

│
└── requirements.txt
```

---

# 🚀 How To Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run scripts/app.py
```

