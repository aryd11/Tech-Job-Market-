import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from trial import JobMarketIntelligence  # ← Import dari trial.py

warnings.filterwarnings('ignore')

# Cukup import dari trial.py saja seperti di atas.

# Page configuration
st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #A0522D !important;  /* Coklat */
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-card {
        background-color: #FFF9C4;  /* Kuning pastel */
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #5D4037;  /* Coklat */
    }

    .stApp {
        background: #F5DEB3;  /* Coklat muda/creamy */
    }

    [data-testid="stSidebar"] {
        background: #DEB887;  /* Coklat muda */
    }

    [data-testid="stSidebar"] * {
        color: #5D4037 !important;  /* Coklat tua */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'jmi' not in st.session_state:
    with st.spinner('🚀 Loading job market data...'):
        st.session_state.jmi = JobMarketIntelligence("cleaned_data.csv")
        st.session_state.loaded = True

jmi = st.session_state.jmi

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/job.png", width=80)
    st.title("🎯 Navigation")

    page = st.radio(
        "Choose a section:",
        ["💡 Personal Recommendations", "🏠 Dashboard", "📈 Skill Analysis", "🎯 Skill Gaps",
         "🚀 Emerging Skills", "⭐ Hiring Priorities"]
    )

    st.divider()

    # Global filters
    st.subheader("🔍 Filters")
    countries = ['All'] + sorted(jmi.df['country'].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries)

    skill_types = ['languages', 'frameworks', 'databases', 'platforms']
    selected_skill_type = st.selectbox("Skill Type", skill_types)

# ==================== PAGE 1: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">📊 Job Market Intelligence Dashboard</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Developers", f"{len(jmi.df):,}")

    with col2:
        unique_countries = jmi.df['country'].nunique()
        st.metric("Countries", unique_countries)

    with col3:
        top_lang = jmi.get_top_skills('languages', 1).index[0]
        st.metric("Most Popular Language", top_lang)

    with col4:
        total_skills = jmi.df['skills_languages'].dropna().str.split(';').explode().nunique()
        st.metric("Unique Skills", total_skills)

    st.divider()

    # Top Job Titles
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💼 Top Job Titles            ")
        top_titles = jmi.get_top_job_titles(10)
        fig, ax = plt.subplots(figsize=(16, 13))
        top_titles.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_xlabel('Number of Developers')
        ax.set_title('Most Common Job Roles')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🛠️ Top Programming Languages")
        if selected_country == 'All':
            top_skills = jmi.get_top_skills(selected_skill_type, 6)
        else:
            top_skills = jmi.get_skills_by_country(selected_country, selected_skill_type, 10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_skills.plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_xlabel('Number of Developers')
        ax.set_title(
            f'Top {selected_skill_type.title()} in {"Global" if selected_country == "All" else selected_country}')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

# ==================== PAGE 2: SKILL ANALYSIS ====================
elif page == "📈 Skill Analysis":
    st.markdown('<div class="main-header">📈 Skill Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"✅ Most Used {selected_skill_type.title()}")
        if selected_country == 'All':
            have_skills = jmi.get_top_skills(selected_skill_type, 15)
        else:
            have_skills = jmi.get_skills_by_country(selected_country, selected_skill_type, 15)

        fig, ax = plt.subplots(figsize=(10, 6))
        have_skills.plot(kind='barh', ax=ax, color='#2E86AB')
        ax.set_xlabel('Number of Developers')
        ax.set_title(f'Most Used {selected_skill_type.title()}')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"🎯 Most Wanted {selected_skill_type.title()}")
        if selected_country == 'All':
            want_skills = jmi.get_wanted_skills(selected_skill_type, 15)
        else:
            # Filter wanted skills by country
            country_data = jmi.df[jmi.df['country'] == selected_country]
            col_map = {
                'languages': 'wanted_languages',
                'frameworks': 'wanted_frameworks',
                'databases': 'wanted_databases',
                'platforms': 'wanted_platforms'
            }
            want_skills = (country_data[col_map[selected_skill_type]]
                           .dropna()
                           .str.split(';')
                           .explode()
                           .str.strip()
                           .str.title()
                           .value_counts()
                           .head(15))

        fig, ax = plt.subplots(figsize=(10, 6))
        want_skills.plot(kind='barh', ax=ax, color='#A23B72')
        ax.set_xlabel('Number of Developers Who Want to Learn')
        ax.set_title(f'Most Wanted {selected_skill_type.title()}')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

# ==================== PAGE 3: SKILL GAPS ====================
elif page == "🎯 Skill Gaps":
    st.markdown('<div class="main-header">🎯 Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Skills that are in high demand but low supply*")

    gaps = jmi.get_skill_gap(selected_skill_type, 15)

    if len(gaps) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(gaps))
            width = 0.35
            ax.bar([i - width / 2 for i in x], gaps['have_count'], width, label='Have', color='#FFB347')
            ax.bar([i + width / 2 for i in x], gaps['want_count'], width, label='Want', color='#FF6B6B')
            ax.set_xlabel('Skill')
            ax.set_ylabel('Number of Developers')
            ax.set_title(f'Top Skill Gaps - {selected_skill_type.title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(gaps['skill'], rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("📊 Gap Summary")
            for _, row in gaps.head(8).iterrows():
                st.markdown(f"""
                <div class="skill-card">
                    <strong>{row['skill']}</strong><br>
                    Want: {row['want_count']:,} | Have: {row['have_count']:,}<br>
                    <span style='color: #FF6B6B;'>Gap: {row['gap']:,}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No significant skill gaps found for this category")

# ==================== PAGE 4: EMERGING SKILLS ====================
elif page == "🚀 Emerging Skills":
    st.markdown('<div class="main-header">🚀 Emerging Skills</div>', unsafe_allow_html=True)
    st.markdown("*Skills with fastest-growing demand (Want/Have ratio)*")

    emerging = jmi.get_emerging_skills(selected_skill_type, 15)

    if len(emerging) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(emerging['skill'], emerging['demand_ratio'], color='#9B59B6')
        ax.set_xlabel('Demand Ratio (Want / Have)')
        ax.set_title(f'Emerging {selected_skill_type.title()} Skills')
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, ratio) in enumerate(zip(bars, emerging['demand_ratio'])):
            ax.text(ratio + 0.05, bar.get_y() + bar.get_height() / 2,
                    f'{ratio:.1f}x', va='center', fontsize=10)

        st.pyplot(fig)
        plt.close()

        # Show as table
        st.subheader("📋 Detailed Analysis")
        st.dataframe(emerging.style.highlight_max(subset=['demand_ratio'], color='#9B59B6'))
    else:
        st.info("No emerging skills identified for this category")

# ==================== PAGE 5: PERSONAL RECOMMENDATIONS ====================
elif page == "💡 Personal Recommendations":
    st.markdown('<div class="main-header">💡 Personalized Skill Recommendations</div>', unsafe_allow_html=True)

    with st.form("user_profile"):
        st.subheader("Tell us about yourself")

        col1, col2 = st.columns(2)

        with col1:
            current_skills = st.text_input(
                "Your current skills (comma-separated)",
                placeholder="e.g., Python, SQL, JavaScript"
            )

            career_goal = st.selectbox(
                "Career goal",
                ["Data Scientist", "Web Developer", "DevOps Engineer",
                 "Mobile Developer", "Machine Learning Engineer", "Other"]
            )

        with col2:
            target_country = st.selectbox("Target country", countries[1:])  # Exclude 'All'

            if career_goal == "Other":
                career_goal = st.text_input("Specify your target role")

        submitted = st.form_submit_button("🎯 Get Recommendations", use_container_width=True)

    if submitted and current_skills:
        skill_list = [s.strip() for s in current_skills.split(',')]

        with st.spinner("Analyzing market data..."):
            recommendations = jmi.recommend_skills(
                skill_list,
                career_goal if career_goal != "Other" else None,
                target_country if target_country != "All" else None
            )

            st.success("✅ Here are your personalized recommendations!")

            # Display recommendations as cards
            if len(recommendations) > 0:
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"## {i}")
                        with col2:
                            demand_score = int(row['demand_score'] / 2) if row['demand_score'] else 0
                            st.markdown(f"""
                            ### 🚀 {row['skill']}
                            **Demand Score:** {'⭐' * demand_score}{'☆' * (5 - demand_score)} ({row['demand_score']}/10)

                            **Why learn this?** {row['reasons']}
                            """)
                        st.divider()
            else:
                st.info("No recommendations found. Try adding different skills.")
    elif submitted:
        st.warning("Please enter your current skills")

# ==================== PAGE 6: HIRING PRIORITIES ====================
elif page == "⭐ Hiring Priorities":
    st.markdown('<div class="main-header">⭐ Hiring Priorities</div>', unsafe_allow_html=True)
    st.markdown("*What employers value most when hiring*")

    country_for_hiring = selected_country if selected_country != 'All' else None

    priorities = jmi.get_hiring_priorities(country_for_hiring)

    if not priorities.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            priorities.plot(kind='bar', ax=ax, color='#E74C3C')
            ax.set_ylabel('Percentage of Employers (%)')
            ax.set_title(f'Hiring Priorities{" in " + selected_country if selected_country != "All" else " (Global)"}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim(0, 100)

            # Add value labels
            for i, (priority, value) in enumerate(priorities.items()):
                ax.text(i, value + 1, f'{value:.0f}%', ha='center', fontsize=10)

            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("📊 Key Insights")
            top_priority = priorities.index[0]
            top_value = priorities.iloc[0]
            st.info(
                f"**Top Priority:** {top_priority.replace('_', ' ').title()}\n\n{top_value:.0f}% of employers consider this important or very important")

            st.subheader("💡 Advice")
            st.markdown(f"""
            - Focus on **{top_priority.replace('_', ' ').title()}** to stand out
            - The lowest priority is **{priorities.index[-1].replace('_', ' ').title()}** ({priorities.iloc[-1]:.0f}%)
            """)
    else:
        st.warning("Hiring priority data not available for this selection")

# Footer
st.divider()
st.markdown("""
<p style='text-align: center; color: gray;'>
    📊 Job Market Intelligence Platform | Powered by Stack Overflow Survey Data
</p>
""", unsafe_allow_html=True)