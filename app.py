import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career Recommendation System",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
GLOBAL BACKGROUND
========================================================= */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {

    background-color:#F5EDED;
}

.stApp {

    background-color:#F5EDED;
}

html, body, [class*="css"] {

    font-family:'Segoe UI', sans-serif;

    color:#111827;
}

/* =========================================================
TYPOGRAPHY
========================================================= */

.section-title {

    font-size:30px;

    font-weight:700;

    margin-top:25px;

    margin-bottom:25px;

    color:#0f172a;
}

/* =========================================================
SUB SECTION TITLES
========================================================= */

.sub-section-title {

    font-size:22px;

    font-weight:500;

    color:#111827;

    margin-top:22px;

    margin-bottom:18px;

    line-height:1.4;
}

.job-title {

    font-size:40px;

    font-weight:900;

    color:#0f172a;
}

.skills-container {

    display:flex;

    flex-wrap:wrap;

    gap:12px;

    margin-top:10px;

    margin-bottom:20px;
}

.content-title {

    font-size:18px;

    font-weight:700;

    color:#111827;

    margin-top:18px;

    margin-bottom:14px;

    margin-left:14px;

    line-height:1.4;
}

/* =========================================================
BADGES
========================================================= */

.rank-badge {

    display:inline-block;

    background:linear-gradient(
    135deg,
    #2563eb,
    #1d4ed8
    );

    color:white;

    padding:10px 18px;

    border-radius:12px;

    font-weight:700;

    margin-bottom:18px;

    box-shadow:0 4px 14px rgba(37,99,235,0.18);
}


/* =========================================================
SKILL PILLS
========================================================= */

.skill-pill-green {

    display:inline-flex;

    align-items:center;

    justify-content:center;

    vertical-align:middle;

    background:#dcfce7;

    color:#166534;

    padding:10px 16px;

    border-radius:20px;

    font-size:14px;

    font-weight:700;

    margin:6px;

    min-height:44px;

    border:1px solid #bbf7d0;
}

.skill-pill-red {

    display:inline-flex;

    align-items:center;

    justify-content:center;

    vertical-align:middle;

    background:#fee2e2;

    color:#991b1b;

    padding:10px 16px;

    border-radius:20px;

    font-size:14px;

    font-weight:700;

    margin:6px;

    min-height:44px;

    border:1px solid #fecaca;
}


/* =========================================================
COLUMN HEIGHT
========================================================= */

div[data-testid="column"] {

    align-self:stretch !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="
background: linear-gradient(135deg, #0f172a, #1e293b);
padding:14px 32px;
border-radius:24px;
text-align:center;
margin-bottom:28px;
box-shadow:0 6px 20px rgba(0,0,0,0.10);
">

<div style="
font-size:36px;
margin-bottom:0px;
">
🚀
</div>

<div style="
font-size:42px;
font-weight:900;
color:white;
line-height:1.05;
margin-bottom:10px;
">

Career Recommendation System

</div>

<div style="
font-size:18px;
color:#cbd5e1;
max-width:850px;
margin:auto;
line-height:1.5;
font-weight:400;
">

Analyze your profile and discover the best matching career opportunities
based on your skills, experience, qualifications and career preferences.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

jobs_df = pd.read_csv("../data/cleaned_jobs_dataset.csv")
tech_df = pd.read_csv("../data/technical_skills.csv")
soft_df = pd.read_csv("../data/soft_skills.csv")

jobs_df = jobs_df.fillna("")

# ============================================================
# SKILL LISTS
# ============================================================

technical_skills_list = (
    tech_df["skill"]
    .dropna()
    .astype(str)
    .str.lower()
    .str.strip()
    .unique()
    .tolist()
)

soft_skills_list = (
    soft_df["skill"]
    .dropna()
    .astype(str)
    .str.lower()
    .str.strip()
    .unique()
    .tolist()
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_skills(value):

    if pd.isna(value) or value == "":
        return set()

    return set([
        skill.strip().lower()
        for skill in str(value).split(",")
        if skill.strip()
    ])

# ============================================================
# MATCH LOGIC FUNCTIONS
# ============================================================

# -------------------------------------------------
# skill match: Technical and Soft Skills
# -------------------------------------------------

def calculate_skill_match(user_skills, required_skills):

    user_set = set([
        skill.lower()
        for skill in user_skills
    ])

    required_set = split_skills(required_skills)

    if len(required_set) == 0:
        return 0, set(), set()

    matched = user_set.intersection(required_set)

    missing = required_set.difference(user_set)

    score = len(matched) / len(required_set)

    return score, matched, missing


# -------------------------------------------------
# location / work_mode / employment_type match
# -------------------------------------------------

def exact_match(user_values, job_value):

    job_value = str(job_value).strip().lower()

    user_values = [

        str(value).strip().lower()

        for value in user_values
    ]

    if "all" in user_values:
        return 1

    if job_value in user_values:
        return 1

    return 0

# -------------------------------------------------
# experience_match
# -------------------------------------------------

def experience_match(user_exp, required_exp):

    try:
        required_exp = float(required_exp)

    except ValueError:
        required_exp = 0

    if required_exp == 0:
        return 1

    if user_exp >= required_exp:
        return 1

    return user_exp / required_exp


# -------------------------------------------------
# education_match
# -------------------------------------------------

def education_match(user_education, job_education):

    education_levels = {

        "bachelor's degree": 1,

        "master's degree": 2,

        "phd": 3
    }

    user_score = education_levels.get(
        str(user_education).strip().lower(),
        0
    )

    job_score = education_levels.get(
        str(job_education).strip().lower(),
        0
    )

    if job_score == 0:
        return 1

    if user_score >= job_score:
        return 1

    return user_score / job_score

# -------------------------------------------------
# language_match
# -------------------------------------------------

def language_match(user_level, job_level, language="english"):

    level_maps = {

        "english": {
            "intermediate": 1,
            "advanced": 2,
            "fluent": 3
        },

        "danish": {
            "basic": 1,
            "intermediate": 2
        }
    }

    levels = level_maps.get(language, {})

    user_score = levels.get(
        str(user_level).strip().lower(),
        0
    )

    job_score = levels.get(
        str(job_level).strip().lower(),
        0
    )

    if job_score == 0:
        return 1

    if user_score >= job_score:
        return 1

    return user_score / job_score

# ============================================================
# FORM SECTION
# ============================================================

with st.form("candidate_form"):

    st.header("👤 Candidate Profile")

    # --------------------------------------------------------
    # 1. TECHNICAL SKILLS
    # --------------------------------------------------------

    technical_skills = st.multiselect(
        "Technical Skills",
        technical_skills_list,
        max_selections=None,
        placeholder="Select technical skills",
        label_visibility="visible"
    )

    # --------------------------------------------------------
    # 2. SOFT SKILLS
    # --------------------------------------------------------

    soft_skills = st.multiselect(
        "Soft Skills",
        soft_skills_list,
        max_selections=None,
        placeholder="Select soft skills",
        label_visibility="visible"
    )

    # --------------------------------------------------------
    # 3. ROLE CATEGORY
    # --------------------------------------------------------

    role_categories = (
        jobs_df["role_category"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    role_categories = sorted(role_categories)

    preferred_roles = st.multiselect(
        "Preferred Career Roles",
        role_categories,
        placeholder="Select preferred career roles"
    )

    # --------------------------------------------------------
    # 4. EXPERIENCE
    # --------------------------------------------------------

    experience = st.slider(
        "Years of Experience",
        0,
        15,
        2
    )

    # --------------------------------------------------------
    # 5. EDUCATION
    # --------------------------------------------------------

    education = st.selectbox(
        "Education Level",
        [
            "Bachelor's Degree",
            "Master's Degree",
            "PhD"
        ]
    )

    # --------------------------------------------------------
    # 6. ENGLISH LEVEL
    # --------------------------------------------------------

    english_level = st.selectbox(
        "English Level",
        [
            "Intermediate",
            "Advanced",
            "Fluent"
        ]
    )

    # --------------------------------------------------------
    # 7. DANISH LEVEL
    # --------------------------------------------------------

    danish_level = st.selectbox(
        "Danish Level",
        [
            "Basic",
            "Intermediate"
        ]
    )

    # --------------------------------------------------------
    # 8. CITY
    # --------------------------------------------------------

    cities = (
        jobs_df["city"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    cities = sorted(cities)

    preferred_locations = st.multiselect(

        "Preferred Locations",

        options=[
            "All"
        ] + cities,

        default=["All"]
    )

    if "All" in preferred_locations and len(preferred_locations) > 1:

        preferred_locations = ["All"]

    # --------------------------------------------------------
    # 9. WORK MODE
    # --------------------------------------------------------

    preferred_work_modes = st.multiselect(

        "Preferred Work Modes",

        options=[
            "All",
            "Remote",
            "Hybrid",
            "On-site"
        ],

        default=["All"]
    )

    if "All" in preferred_work_modes and len(preferred_work_modes) > 1:

        preferred_work_modes = ["All"]

    # --------------------------------------------------------
    # 10. EMPLOYMENT TYPE
    # --------------------------------------------------------

    preferred_employment_types = st.multiselect(

        "Preferred Employment Types",

        options=[
            "All",
            "Full-time",
            "Part-time",
            "Internship"
        ],

        default=["All"]
    )

    if "All" in preferred_employment_types and len(preferred_employment_types) > 1:

        preferred_employment_types = ["All"]

    # --------------------------------------------------------
    # SUBMIT BUTTON
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "🚀 Analyze My Profile",
        use_container_width=True
    )


# ============================================================
# CAREER Recommendation Engine
# ============================================================

if submitted:

    if not technical_skills:

        st.warning(
            "⚠️ Please select at least one technical skill."
        )

        st.stop()

    if not soft_skills:

        st.warning(
            "⚠️ Please select at least one soft skill."
        )

        st.stop()

    if not preferred_roles:

        st.warning(
            "⚠️ Please select at least one preferred career role."
        )

        st.stop()
 
    # ---------------------------------------------------------=
    # CANDIDATE SUMMARY
    # ----------------------------------------------------------

    st.markdown("""
    <hr style="
    margin-top:40px;
    margin-bottom:40px;
    border:none;
    border-top:2px solid rgba(0,0,0,0.12);
    ">
    """, unsafe_allow_html=True)

    preferred_roles_text = (
        ", ".join(preferred_roles)
        if preferred_roles
        else "multiple career opportunities"
    )

    technical_skills_text = (
        ", ".join(technical_skills)
        if technical_skills
        else "No technical skills selected"
    )

    soft_skills_text = (
        ", ".join(soft_skills)
        if soft_skills
        else "No soft skills selected"
    )


    display_locations = (

        "All Locations"

        if "All" in preferred_locations

        else ", ".join(preferred_locations)
    )

    display_work_modes = (

        "All (Remote, Hybrid, On-site)"

        if "All" in preferred_work_modes

        else ", ".join(preferred_work_modes)
    )

    display_employment_types = (

        "All (Full-time, Part-time, Internship)"

        if "All" in preferred_employment_types

        else ", ".join(preferred_employment_types)
    )

       
    summary_html = f"""

    <div style="
    background:white;
    padding:40px;
    border-radius:28px;
    box-shadow:0 8px 24px rgba(0,0,0,0.06);
    line-height:2.1;
    font-size:19px;
    color:#374151;
    margin-bottom:35px;
    ">

    <div style="
    font-size:34px;
    font-weight:900;
    color:#111827;
    margin-bottom:28px;
    ">

    Your Professional Profile Summary

    </div>

    <p>

    You are a candidate with a
    <b>{education}</b>
    and approximately
    <b>{experience} years</b>
    of professional experience.

    </p>

    <p>

    Your primary career focus is currently on
    <b>{preferred_roles_text}</b>
    opportunities.

    </p>

    <p>

    You prefer working in
    <b>{display_locations}</b>
    with a
    <b>{display_work_modes}</b>
    work setup and
    <b>{display_employment_types}</b>
    employment type.

    </p>

    <p>

    Your technical skill set includes
    <b>{technical_skills_text}</b>.

    </p>

    <p>

    On the soft skills side, you are confident in
    <b>{soft_skills_text}</b>.

    </p>

    <p>

    Your English level is
    <b>{english_level}</b>
    and your Danish level is
    <b>{danish_level}</b>.

    </p>

    <p>

    You are looking for opportunities where you can continue growing,
    contribute to impactful projects, and help drive real business value.

    </p>

    </div>
    """

    st.markdown(summary_html, unsafe_allow_html=True)

    # FILTER JOBS
    filtered_jobs_df = jobs_df.copy()

    if preferred_roles:
        filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["role_category"].isin(preferred_roles)]
    
# BUILD JOB PROFILE TEXT
    filtered_jobs_df["job_profile"] = (

        filtered_jobs_df["technical_skills_required"]
        .astype(str)

        + " " +

        filtered_jobs_df["soft_skills_required"]
        .astype(str)

        + " " +

        filtered_jobs_df["role_category"]
        .astype(str)
    ).str.lower()
    
    if filtered_jobs_df.empty:
        st.warning("No matching jobs found for selected role categories.")
        st.stop()
    
# BUILD USER PROFILE TEXT
    user_profile = " ".join(

        technical_skills +

        soft_skills +

        preferred_roles
    ).lower()
        
# TF-IDF VECTORIZATION
    tfidf = TfidfVectorizer()

    job_vectors = tfidf.fit_transform(
        filtered_jobs_df["job_profile"]
    )

    user_vector = tfidf.transform(
        [user_profile]
    )

# COSINE SIMILARITY
    similarity_scores = cosine_similarity(

        user_vector,

        job_vectors

    ).flatten()

    # ----------------------------------------------------------
    # Match Calculation
    # ----------------------------------------------------------
    
    results = []

    for idx, (_, job) in enumerate(filtered_jobs_df.iterrows()):
        similarity_score = similarity_scores[idx]
        technical_score, matched_tech, missing_tech = (
            calculate_skill_match(
                technical_skills,
                job["technical_skills_required"]
            )
        )

        soft_score, matched_soft, missing_soft = (
            calculate_skill_match(
                soft_skills,
                job["soft_skills_required"]
            )
        )

        exp_score = experience_match(
            experience,
            job["experience_required_years"]
        )

        location_score = exact_match(

            preferred_locations,

            job["city"]
        )
        
        work_mode_score = exact_match(

            preferred_work_modes,

            job["work_mode"]
        )
        
        employment_score = exact_match(

            preferred_employment_types,

            job["employment_type"]
        )

        education_score = education_match(
            education,
            job["education_required"]
        )

        english_score = language_match(
            english_level,
            job["English_required_level"],
            language="english"
        )

        danish_score = language_match(
            danish_level,
            job["Danish_required_level"],
            language="danish"
        )
        
        # final_score = (

        #     technical_score * 0.45 +

        #     soft_score * 0.15 +

        #     exp_score * 0.10 +

        #     education_score * 0.08 +

        #     english_score * 0.03 +

        #     danish_score * 0.04 +

        #     location_score * 0.05 +

        #     work_mode_score * 0.05 +

        #     employment_score * 0.05
        # )

        final_score = (

            similarity_score * 0.35 +    # Calculated using TF-IDF vectorization  and Cosine Similarity between user profile and job profile

            technical_score * 0.20 +     # Exact technical skill matching (intersection-based)

            soft_score * 0.10 +          # Exact soft skill matching (intersection-based)

            exp_score * 0.10 +

            education_score * 0.08 +

            english_score * 0.05 +

            danish_score * 0.05 +

            location_score * 0.03 +

            work_mode_score * 0.02 +

            employment_score * 0.02
        )


        results.append({

            "Company":
                job["company_name"],

            "Job Title":
                job["job_title"],

            "City":
                job["city"],

            "Work Mode":
                job["work_mode"],

            "Employment Type":
                job["employment_type"],

            "Final Match Score":
                round(final_score * 100, 2),

            "Technical Match":
                round(technical_score * 100, 2),

            "Soft Skill Match":
                round(soft_score * 100, 2),

            "Experience Match":
                round(exp_score * 100, 2),

            "Education Match":
                round(education_score * 100, 2),

            "English Match":
                round(english_score * 100, 2),

            "Danish Match":
                round(danish_score * 100, 2),

            "Location Match":
                round(location_score * 100, 2),

            "Work Mode Match":
                round(work_mode_score * 100, 2),

            "Employment Match":
                round(employment_score * 100, 2),

            "Matched Technical":
                sorted(matched_tech),

            "Missing Technical":
                sorted(missing_tech),

            "Matched Soft":
                sorted(matched_soft),

            "Missing Soft":
                sorted(missing_soft),
            
            "Semantic Similarity":
                round(similarity_score * 100, 2),
        })

    # ----------------------------------------------------------
    # Recommendations DataFrame
    # ----------------------------------------------------------

    recommendations_df = pd.DataFrame(results)

    recommendations_df = (
        recommendations_df
        .sort_values(
            by="Final Match Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.divider()

    # ----------------------------------------------------------
    # Intro Section
    # ----------------------------------------------------------

    st.markdown("""
<div style="
font-size:20px;
color:#4b5563;
line-height:1.8;
max-width:1100px;
">

I analyzed your background, skills, experience,
work preferences, and career interests
to find the strongest matches for you.

<div style="margin-top:18px;">

I’ll also show you:

<ul style="
margin-top:10px;
padding-left:28px;
line-height:2;
">

<li>where your profile is already strong</li>

<li>where you still have skill gaps</li>

<li>what you can improve</li>

<li>which opportunities fit you best right now</li>

</ul>

</div>

</div>
""", unsafe_allow_html=True)

    # ----------------------------------------------------------
    # Top Jobs Selection
    # ----------------------------------------------------------

    top_jobs = recommendations_df.head(5)

    for index, row in top_jobs.iterrows():

        st.markdown(f"""
    <div class="rank-badge">
    ATS Rank #{index + 1}
    </div>
    """, unsafe_allow_html=True)
        
        with st.expander(
            f"🥇 • {row['Job Title']} • {row['Final Match Score']}%",
            expanded=(index == 0)
        ):
        
        
        # with st.expander(
        #     f"#{index + 1} {row['Job Title']} ({row['Final Match Score']}%)",
        #     expanded=(index == 0)
        # ):

            # --------------------------------------------------
            # Rank Badge
            # --------------------------------------------------


            # --------------------------------------------------
            # Job Title
            # --------------------------------------------------

            st.markdown(f"""
<div class="job-title">
{row['Job Title']}
</div>
""", unsafe_allow_html=True)

            # --------------------------------------------------
            # Job Caption
            # --------------------------------------------------

            st.caption(
                f"{row['Company']} • "
                f"{row['City']} • "
                f"{row['Work Mode']} • "
                f"{row['Employment Type']}"
            )

            # --------------------------------------------------
            # MATCH ANALYSIS
            # --------------------------------------------------

            st.markdown("""
<div class="section-title">
📊 Match Analysis
</div>
""", unsafe_allow_html=True)

            scores = [

                ("🟢 Final Match", row['Final Match Score'], "#22c55e"),
                
                ("🔵 Technical Match", row['Technical Match'], "#3b82f6"),

                ("🟣 Soft Skill Match", row['Soft Skill Match'], "#8b5cf6"),
            ]

            for label, value, color in scores:

                font_size = "38px" if "Final" in label else "18px"

                bar_height = "22px" if "Final" in label else "16px"

                st.markdown(f"""
<div style="
margin-bottom:28px;
">

<div style="
font-size:{font_size};
font-weight:900;
color:{color};
margin-bottom:10px;
">

{label}: {value}%

</div>

<div style="
width:100%;
height:{bar_height};
background:#e5e7eb;
border-radius:12px;
overflow:hidden;
">

<div style="
width:{value}%;
height:{bar_height};
background:{color};
border-radius:12px;
">
</div>

</div>

</div>
""", unsafe_allow_html=True)

            # ==================================================
            # SMALL METRICS
            # ==================================================

            metric_col1, metric_col2 = st.columns(2)

            # --------------------------------------------------
            # Experience Match
            # --------------------------------------------------

            with metric_col1:

                st.markdown(f"""
<div style="
background:white;
padding:10px 22px;
border-radius:16px;
margin-top:10px;
margin-bottom:18px;
border-left:5px solid #f97316;
box-shadow:0 2px 10px rgba(0,0,0,0.05);
">

<div style="
font-size:14px;
font-weight:700;
color:#111827;
margin-bottom:8px;
">

🟠 Experience Match

</div>

<div style="
font-size:28px;
font-weight:900;
color:#0f172a;
">

{row['Experience Match']}%

</div>

</div>
""", unsafe_allow_html=True)

            # --------------------------------------------------
            # Education Match
            # --------------------------------------------------

            with metric_col2:

                st.markdown(f"""
<div style="
background:white;
padding:10px 22px;
border-radius:16px;
margin-top:10px;
margin-bottom:18px;
border-left:5px solid #14b8a6;
box-shadow:0 2px 10px rgba(0,0,0,0.05);
">

<div style="
font-size:14px;
font-weight:800;
color:#111827;
margin-bottom:8px;
">

🎓 Education Match

</div>

<div style="
font-size:28px;
font-weight:900;
color:#0f172a;
">

{row['Education Match']}%

</div>

</div>
""", unsafe_allow_html=True)

            # ==================================================
            # BADGES
            # ==================================================

            badge_html = ""

            if row['Location Match'] >= 100:

                badge_html += """
<span class="skill-pill-green">
📍 Location Match
</span>
"""

            if row['Work Mode Match'] >= 100:

                badge_html += """
<span class="skill-pill-green">
🏠 Work Mode Match
</span>
"""

            if row['English Match'] >= 100:

                badge_html += """
<span class="skill-pill-green">
🗣️ English Match
</span>
"""

            if row['Danish Match'] >= 100:

                badge_html += """
<span class="skill-pill-green">
🗣️ Danish Match
</span>
"""

            if row['Employment Match'] >= 100:

                badge_html += """
<span class="skill-pill-green">
💼 Employment Match
</span>
"""

            # --------------------------------------------------
            # Render Badges
            # --------------------------------------------------

            st.markdown(f"""
<div style="
margin-top:18px;
margin-bottom:28px;
display:flex;
flex-wrap:wrap;
gap:12px;
align-items:center;
">

{badge_html}

</div>
""", unsafe_allow_html=True)
            

            # ==========================================================
            # SKILL GAP ANALYSIS
            # ==========================================================

            st.markdown("""
            <hr style="
            margin-top:60px;
            margin-bottom:60px;
            border:none;
            border-top:2px solid rgba(0,0,0,0.12);
            ">
            """, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">🧠 Skill Gap Analysis</div>',
                unsafe_allow_html=True
            )

            # ==========================================================
            # TECHNICAL SKILL GAP ANALYSIS
            # ==========================================================

            st.markdown(
                '<div class="sub-section-title">💻 Technical Skill Gap Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="content-title">✅ Matched Technical Skills</div>',
                unsafe_allow_html=True
            )

            if row["Matched Technical"]:

                matched_html = ""

                for skill in row["Matched Technical"]:

                    matched_html += f"""
            <span class="skill-pill-green">
            {skill}
            </span>
            """

                st.markdown(f"""
            <div class="skills-container">

            {matched_html}

            </div>
            """, unsafe_allow_html=True)

            else:

                st.info("No matched technical skills.")

            st.markdown(
                '<div class="content-title">❌ Missing Technical Skills</div>',
                unsafe_allow_html=True
            )

            if row["Missing Technical"]:

                missing_html = ""

                for skill in row["Missing Technical"]:

                    missing_html += f"""
            <span class="skill-pill-red">
            {skill}
            </span>
            """

                st.markdown(f"""
            <div class="skills-container">

            {missing_html}

            </div>
            """, unsafe_allow_html=True)

            else:

                st.success("No missing technical skills.")
            # ==========================================================
            # SOFT SKILL GAP ANALYSIS
            # ==========================================================

            st.markdown(
                '<div class="sub-section-title">🧠 Soft Skill Gap Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="content-title">✅ Matched Soft Skills</div>',
                unsafe_allow_html=True
            )

            if row["Matched Soft"]:

                soft_match_html = ""

                for skill in row["Matched Soft"]:

                    soft_match_html += f"""
            <span class="skill-pill-green">
            {skill}
            </span>
            """

                st.markdown(f"""
            <div class="skills-container">

            {soft_match_html}

            </div>
            """, unsafe_allow_html=True)

            else:

                st.info("No matched soft skills.")

            st.markdown(
                '<div class="content-title">❌ Missing Soft Skills</div>',
                unsafe_allow_html=True
            )

            if row["Missing Soft"]:

                soft_missing_html = ""

                for skill in row["Missing Soft"]:

                    soft_missing_html += f"""
            <span class="skill-pill-red">
            {skill}
            </span>
            """

                st.markdown(f"""
            <div class="skills-container">

            {soft_missing_html}

            </div>
            """, unsafe_allow_html=True)

            else:

                st.success("No missing soft skills.")

            # ==========================================================
            # SUCCESS ROADMAP
            # ==========================================================

            st.markdown("""
    <hr style="
    margin-top:50px;
    margin-bottom:45px;
    border:none;
    border-top:2px solid rgba(0,0,0,0.12);
    ">
    """, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">🚀 Personalized Success Roadmap</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
    <div style="
    font-size:18px;
    line-height:2;
    color:#475569;
    margin-bottom:35px;
    ">

    Based on your current profile and skill gaps, here are some personalized recommendations
    that can help you become a stronger match for this opportunity.

    </div>
    """, unsafe_allow_html=True)

            # ==========================================================
            # TECHNICAL SKILL RECOMMENDATIONS
            # ==========================================================

            technical_skill_recommendations = {

                "sql":
                    "Practice advanced SQL queries and real-world analytics",

                "sql basics":
                    "Strengthen SQL fundamentals with hands-on exercises",

                "excel":
                    "Improve Excel reporting and analytical workflows",

                "advanced excel":
                    "Improve advanced Excel analytics and automation skills",

                "power bi":
                    "Build interactive Power BI dashboards and KPIs",

                "power query":
                    "Practice data transformation with Power Query",

                "tableau":
                    "Improve storytelling and dashboard visualization skills",

                "reporting":
                    "Practice business reporting and insight generation",

                "dashboard optimization":
                    "Improve dashboard performance and UX design",

                "visualization":
                    "Improve data visualization and storytelling techniques",

                "data storytelling":
                    "Improve storytelling with data and visual insights",

                "dax":
                    "Practice advanced DAX measures and calculations",

                "google analytics":
                    "Practice web and customer analytics with Google Analytics",

                "python":
                    "Build real-world Python data projects",

                "python basics":
                    "Strengthen Python fundamentals with practical exercises",

                "pandas":
                    "Practice data cleaning and transformation with Pandas",

                "numpy":
                    "Improve analytical programming with NumPy",

                "matplotlib":
                    "Improve data visualization with Matplotlib",

                "seaborn":
                    "Practice advanced data storytelling with Seaborn",

                "etl pipeline design":
                    "Build scalable ETL pipeline projects",

                "data pipelines":
                    "Build scalable production-style data pipelines",

                "airflow":
                    "Learn workflow orchestration with Apache Airflow",

                "spark":
                    "Practice big data processing with PySpark",

                "kafka":
                    "Practice event streaming and real-time data pipelines",

                "apache kafka":
                    "Learn real-time data streaming with Apache Kafka",

                "data warehousing":
                    "Improve dimensional modeling and warehousing concepts",

                "data modeling":
                    "Practice database and analytical data modeling",

                "machine learning":
                    "Build practical machine learning projects",

                "tensorflow":
                    "Build deep learning projects using TensorFlow",

                "pytorch":
                    "Practice neural network development with PyTorch",

                "scikit-learn":
                    "Improve classical machine learning workflows",

                "deep learning":
                    "Build deep learning and neural network projects",

                "computer vision":
                    "Build computer vision and image-processing projects",

                "nlp":
                    "Build natural language processing applications",

                "mlops":
                    "Practice MLOps workflows and ML deployment pipelines",

                "docker":
                    "Build containerized applications with Docker",

                "docker basics":
                    "Move beyond Docker basics with hands-on container projects",

                "kubernetes":
                    "Practice container orchestration with Kubernetes",

                "terraform":
                    "Build Infrastructure as Code projects using Terraform",

                "aws":
                    "Improve AWS cloud and data engineering skills",

                "azure":
                    "Practice cloud architecture and deployment on Azure",

                "cloud security":
                    "Improve cloud security and infrastructure protection skills",

                "linux":
                    "Practice Linux command-line and server management skills",

                "git":
                    "Improve Git version control and collaboration workflows",

                "jenkins":
                    "Practice CI/CD automation with Jenkins",

                "ci/cd":
                    "Build CI/CD workflows and deployment automation",

                "streamlit":
                    "Create interactive data applications with Streamlit",

                "figma":
                    "Practice UI/UX design and prototyping with Figma",

                "wireframing":
                    "Improve low-fidelity and high-fidelity wireframing skills"
            }

            # ==========================================================
            # SOFT SKILL RECOMMENDATIONS
            # ==========================================================

            soft_skill_recommendations = {

                "problem solving":
                    "Practice analytical and problem-solving scenarios",

                "stakeholder communication":
                    "Improve stakeholder communication and business alignment",

                "adaptability":
                    "Practice working effectively in fast-changing environments",

                "critical thinking":
                    "Strengthen critical thinking and decision-making skills",

                "analytical thinking":
                    "Improve analytical reasoning and business analysis skills",

                "attention to detail":
                    "Practice delivering accurate and detail-oriented work",

                "teamwork":
                    "Improve collaboration and teamwork experience",

                "leadership":
                    "Develop leadership and project ownership skills",

                "presentation skills":
                    "Practice public speaking and presentation delivery",

                "time management":
                    "Practice prioritization and time management skills",

                "creativity":
                    "Practice creative problem solving and ideation",

                "project management":
                    "Improve planning and project execution skills",

                "communication":
                    "Improve professional communication and collaboration skills"
            }

            # ==========================================================
            # DYNAMIC ROADMAP GENERATION
            # ==========================================================

            technical_roadmap = []

            soft_roadmap = []

            for skill in row["Missing Technical"]:

                skill_lower = skill.lower()

                recommendation = technical_skill_recommendations.get(

                    skill_lower,

                    f"Strengthen your {skill} skills with practical real-world projects and hands-on experience"
                )

                technical_roadmap.append(recommendation)

            for skill in row["Missing Soft"]:

                skill_lower = skill.lower()

                recommendation = soft_skill_recommendations.get(

                    skill_lower,

                    f"Continue improving your {skill} skills through teamwork, collaboration, and practical experience"
                )

                soft_roadmap.append(recommendation)

            # ==========================================================
            # FALLBACKS
            # ==========================================================

            if len(technical_roadmap) == 0:

                technical_roadmap = [
                    "Your technical skills already match this opportunity very well"
                ]

            if len(soft_roadmap) == 0:

                soft_roadmap = [
                    "Your soft skills already align strongly with this role"
                ]

            # ==========================================================
            # DYNAMIC MATCH RECOMMENDATION FUNCTIONS
            # ==========================================================

            def location_recommendations(score):

                if score >= 95:

                    return [
                        "Your preferred location already aligns very well with this role"
                    ]

                elif score >= 70:

                    return [
                        "Stay flexible with nearby opportunities",
                        "Consider hybrid work arrangements"
                    ]

                else:

                    return [
                        "Expand your job search flexibility",
                        "Consider relocation or remote opportunities",
                        "Explore additional nearby job markets"
                    ]


            def employment_recommendations(score):

                if score >= 95:

                    return [
                        "Your employment preference aligns strongly with this opportunity"
                    ]

                elif score >= 70:

                    return [
                        "Stay open to flexible work arrangements",
                        "Explore similar employment models"
                    ]

                else:

                    return [
                        "Consider broader employment opportunities",
                        "Explore internship or flexible work options",
                        "Stay open to alternative work arrangements"
                    ]


            def education_recommendations(score):

                if score >= 95:

                    return [
                        "Your educational background already matches this role very well"
                    ]

                elif score >= 70:

                    return [
                        "Continue strengthening practical industry experience",
                        "Consider additional certifications"
                    ]

                else:

                    return [
                        "Explore certifications and advanced learning paths",
                        "Strengthen practical project experience",
                        "Continue improving technical specialization"
                    ]


            def experience_recommendations(score):

                if score >= 95:

                    return [
                        "Your experience level strongly aligns with this opportunity"
                    ]

                elif score >= 70:

                    return [
                        "Continue building practical project experience",
                        "Strengthen real-world business exposure"
                    ]

                else:

                    return [
                        "Build more hands-on projects",
                        "Gain additional real-world experience",
                        "Contribute to practical business case studies"
                    ]


            def english_recommendations(score):

                if score >= 95:

                    return [
                        "Your English communication already aligns very well with this role"
                    ]

                elif score >= 70:

                    return [
                        "Continue improving professional English communication",
                        "Practice technical interview conversations"
                    ]

                else:

                    return [
                        "Improve business English communication",
                        "Practice technical interview conversations",
                        "Strengthen workplace English fluency"
                    ]


            def danish_recommendations(score):

                if score >= 95:

                    return [
                        "Your Danish communication already aligns well with this role"
                    ]

                elif score >= 70:

                    return [
                        "Continue improving workplace Danish communication",
                        "Practice professional Danish conversations"
                    ]

                else:

                    return [
                        "Strengthen professional Danish vocabulary",
                        "Practice workplace Danish conversations",
                        "Improve daily Danish communication skills"
                    ]

            # ==========================================================
            # ROADMAP SECTIONS
            # ==========================================================

            roadmap_sections = [

                (
                    "💻 Technical Skills Improvements",
                    row['Technical Match'],
                    technical_roadmap
                ),

                (
                    "👥 Soft Skills Improvements",
                    row['Soft Skill Match'],
                    soft_roadmap
                ),

                (
                    "📍 Location Improvements",
                    row['Location Match'],
                    location_recommendations(row['Location Match'])
                ),

                (
                    "💼 Employment Recommendations",
                    row['Employment Match'],
                    employment_recommendations(row['Employment Match'])
                ),

                (
                    "🎓 Education Recommendations",
                    row['Education Match'],
                    education_recommendations(row['Education Match'])
                ),

                (
                    "🧠 Experience Recommendations",
                    row['Experience Match'],
                    experience_recommendations(row['Experience Match'])
                ),

                (
                    "🗣️ English Recommendations",
                    row['English Match'],
                    english_recommendations(row['English Match'])
                ),

                (
                    "🌍 Danish Recommendations",
                    row['Danish Match'],
                    danish_recommendations(row['Danish Match'])
                )
            ]

            # ==========================================================
            # SHOW ONLY IMPROVEMENT AREAS
            # ==========================================================

            roadmap_sections = [

                section

                for section in roadmap_sections

                if section[1] < 95
            ]

            # ==========================================================
            # RENDER ROADMAP
            # ==========================================================

            for title, score, recommendations in roadmap_sections:

                section_title_html = f"""
    <div style="
    margin-top:30px;
    margin-bottom:14px;
    font-size:28px;
    font-weight:900;
    color:#111827;
    ">

    {title} — {score}%

    </div>
    """

                st.markdown(
                    section_title_html,
                    unsafe_allow_html=True
                )

                for item in recommendations:

                    recommendation_html = f"""
    <div style="
    margin-left:18px;
    margin-bottom:10px;
    font-size:18px;
    line-height:1.9;
    color:#374151;
    ">

    • {item}

    </div>
    """

                    st.markdown(
                        recommendation_html,
                        unsafe_allow_html=True
                    )

                st.markdown("""
    <hr style="
    margin-top:25px;
    margin-bottom:25px;
    border:none;
    border-top:1px solid rgba(0,0,0,0.08);
    ">
    """, unsafe_allow_html=True)




            # ==================================================
            # CAREER INSIGHTS EXPLORER
            # ==================================================

            st.markdown("""
            <hr style="
            margin-top:60px;
            margin-bottom:60px;
            border:none;
            border-top:2px solid rgba(0,0,0,0.12);
            ">
            """, unsafe_allow_html=True)
            
            st.markdown(
                '<div class="section-title">📈 Career Insights Explorer</div>',
                unsafe_allow_html=True
            )

            # ==================================================
            # FIRST ROW
            # ==================================================

            viz_col1, viz_col2 = st.columns(2)

            # ==================================================
            # RADAR CHART
            # ==================================================

            with viz_col1:

                radar_categories = [
                    "Technical",
                    "Soft Skills",
                    "Experience",
                    "Education",
                    "Location",
                    "Work Mode",
                    "Employment"
                ]

                radar_values = [
                    row['Technical Match'],
                    row['Soft Skill Match'],
                    row['Experience Match'],
                    row['Education Match'],
                    row['Location Match'],
                    row['Work Mode Match'],
                    row['Employment Match']
                ]

                radar_categories.append(radar_categories[0])
                radar_values.append(radar_values[0])

                radar_fig = go.Figure()

                radar_fig.add_trace(go.Scatterpolar(

                    r=radar_values,
                    theta=radar_categories,
                    fill='toself',

                    fillcolor='rgba(37,99,235,0.35)',

                    line=dict(
                        color='#2563eb',
                        width=3
                    )
                ))

                radar_fig.update_layout(

                    title="🧠 Candidate Match Radar",

                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),

                    showlegend=False,
                    height=450
                )

                st.plotly_chart(
                    radar_fig,
                    use_container_width=True,
                    key=f"radar_{index}_{row['Job Title']}"
                )

            # ==================================================
            # GAUGE CHART
            # ==================================================

            with viz_col2:

                gauge_fig = go.Figure(go.Indicator(

                    mode="gauge+number",

                    value=row['Final Match Score'],

                    title={
                        'text': "🎯 Career Readiness Score"
                    },

                    gauge={

                        'axis': {
                            'range': [0, 100]
                        },

                        'bar': {
                            'color': "#2563eb"
                        },

                        'steps': [

                            {
                                'range': [0, 40],
                                'color': "#fee2e2"
                            },

                            {
                                'range': [40, 70],
                                'color': "#fef3c7"
                            },

                            {
                                'range': [70, 100],
                                'color': "#dcfce7"
                            }
                        ]
                    }
                ))

                gauge_fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True,
                    key=f"gauge_{index}_{row['Job Title']}"
                )
            # ==================================================
            # SECOND ROW
            # ==================================================

            viz_col3, viz_col4 = st.columns(2)

            # ==================================================
            # DONUT CHART
            # ==================================================

            with viz_col3:

                matched_count = (
                    len(row["Matched Technical"]) +
                    len(row["Matched Soft"])
                )

                missing_count = (
                    len(row["Missing Technical"]) +
                    len(row["Missing Soft"])
                )

                donut_fig = go.Figure(data=[go.Pie(

                    labels=[
                        "Matched Skills",
                        "Missing Skills"
                    ],

                    values=[
                        matched_count,
                        missing_count
                    ],

                    hole=0.65,

                    marker=dict(
                        colors=[
                            "#2563eb",
                            "#93c5fd"
                        ]
                    )
                )])

                donut_fig.update_layout(

                    title="🍩 Skills Distribution",

                    height=420
                )

                st.plotly_chart(
                    donut_fig,
                    use_container_width=True,
                    key=f"donut_{index}_{row['Job Title']}"
                )

            # ==================================================
            # CATEGORY SCORE BAR CHART
            # ==================================================

            with viz_col4:

                category_df = pd.DataFrame({

                    "Category": [
                        "Technical",
                        "Soft Skills",
                        "Experience",
                        "Education",
                        "Location",
                        "Work Mode",
                        "Employment"
                    ],

                    "Score": [

                        row['Technical Match'],
                        row['Soft Skill Match'],
                        row['Experience Match'],
                        row['Education Match'],
                        row['Location Match'],
                        row['Work Mode Match'],
                        row['Employment Match']
                    ]
                })

                category_fig = px.bar(

                    category_df,

                    x="Category",

                    y="Score",

                    text="Score",

                    title="📊 Match Category Scores"
                )

                category_fig.update_traces(

                    texttemplate='%{text:.0f}%',

                    textposition='outside'
                )

                category_fig.update_layout(

                    yaxis_range=[0, 110],

                    height=420
                )
                st.plotly_chart(
                    category_fig,
                    use_container_width=True,
                    key=f"category_{index}_{row['Job Title']}"
                )

            # ==================================================
            # TOP MISSING SKILLS
            # ==================================================

            all_missing_skills = (
                row["Missing Technical"] +
                row["Missing Soft"]
            )

            if len(all_missing_skills) > 0:

                missing_df = pd.DataFrame({

                    "Skill": all_missing_skills,

                    "Priority": [
                        len(all_missing_skills) - i
                        for i in range(len(all_missing_skills))
                    ]
                })

                missing_fig = px.bar(

                    missing_df,

                    x="Priority",

                    y="Skill",

                    orientation="h",

                    text="Priority",

                    title="🚨 Top Missing Skills Priority"
                )

                missing_fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    missing_fig,
                    use_container_width=True,
                    key=f"missing_{index}_{row['Job Title']}"
                )