import os
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class JobMarketIntelligence:
    """
    Job Market Intelligence Platform
    Analyzes job market data to identify trends, in-demand skills, and provide recommendations
    """

    # ==================== SKILL WHITELIST PER JOB TITLE ====================
    
    RELEVANT_SKILLS_BY_JOB = {
        "Data Scientist": {
            'Python', 'Sql', 'R', 'Julia', 'Matlab', 'Scala', 'C++'
        },
        "Web Developer": {
            'Javascript', 'Typescript', 'Python', 'Php', 'Ruby', 'C#'
        },
        "Mobile Developer": {
            'Swift', 'Kotlin', 'Java', 'Dart', 'Objective-C'
        },
        "DevOps Engineer": {
            'Python', 'Go', 'Ruby', 'Groovy', 'Perl'
        },
        "Machine Learning Engineer": {
            'Python', 'C++', 'Java', 'Scala', 'R', 'Julia', 'Go'
        }
    }
    

    def __init__(self, data_path: str = "cleaned_data.csv"):
        """
        Initialize the platform with survey data
        """
        print("🚀 Initializing Job Market Intelligence Platform...")

        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            print(f"📍 Current directory: {os.getcwd()}")
            print("📁 Files in current directory:")
            for f in os.listdir():
                print(f"   - {f}")
            raise FileNotFoundError(f"Cannot find {data_path}. Please ensure the file is in the same folder.")

        self.df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.df):,} developer responses")
        
        # Get existing job titles from data
        self.existing_job_titles = set(self.df['job_title'].unique())

    # ==================== HELPER METHODS ====================
    
    def get_matching_job_title(self, career_goal: str) -> Optional[str]:
        """Find matching job title in dataset (case insensitive)"""
        career_lower = career_goal.lower()
        
        # Prioritaskan Data Scientist
        if 'data scientist' in career_lower:
            return 'Data Scientist'
        elif 'web developer' in career_lower:
            return 'Web Developer'
        elif 'mobile developer' in career_lower:
            return 'Mobile Developer'
        elif 'devops engineer' in career_lower:
            return 'DevOps Engineer'
        elif 'machine learning engineer' in career_lower:
            return 'Machine Learning Engineer'
        
        for job in self.existing_job_titles:
            if job.lower() == career_lower:
                return job
            if career_lower in job.lower() or job.lower() in career_lower:
                return job
        
        return None

    # ==================== SKILL ANALYSIS METHODS ====================

    def get_top_skills(self, skill_type: str = 'languages', n: int = 10) -> pd.Series:
        """
        Get top N most common skills
        """
        col_map = {
            'languages': 'skills_languages',
            'frameworks': 'skills_frameworks',
            'databases': 'skills_databases',
            'platforms': 'skills_platforms'
        }

        if skill_type not in col_map:
            raise ValueError(f"Invalid skill_type. Choose from {list(col_map.keys())}")

        col = col_map[skill_type]

        return (self.df[col]
                .dropna()
                .str.split(';')
                .explode()
                .str.strip()
                .str.title()
                .pipe(lambda x: x[x != ''])
                .value_counts()
                .head(n))

    def get_wanted_skills(self, skill_type: str = 'languages', n: int = 10) -> pd.Series:
        """
        Get top N most wanted skills (skills developers want to learn)
        """
        col_map = {
            'languages': 'wanted_languages',
            'frameworks': 'wanted_frameworks',
            'databases': 'wanted_databases',
            'platforms': 'wanted_platforms'
        }

        if skill_type not in col_map:
            raise ValueError(f"Invalid skill_type. Choose from {list(col_map.keys())}")

        col = col_map[skill_type]

        return (self.df[col]
                .dropna()
                .str.split(';')
                .explode()
                .str.strip()
                .str.title()
                .pipe(lambda x: x[x != ''])
                .value_counts()
                .head(n))

    def get_top_skills_by_country(self, country: str, skill_type: str = 'languages', n: int = 100) -> pd.Series:
        """Get top skills ONLY for specific country"""
        country_df = self.df[self.df['country'] == country]
        
        if country_df.empty:
            return pd.Series()
        
        col_map = {
            'languages': 'skills_languages',
            'frameworks': 'skills_frameworks',
            'databases': 'skills_databases',
            'platforms': 'skills_platforms'
        }
        
        col = col_map.get(skill_type, 'skills_languages')
        
        return (country_df[col]
                .dropna()
                .str.split(';')
                .explode()
                .str.strip()
                .str.title()
                .pipe(lambda x: x[x != ''])
                .value_counts()
                .head(n))

    def get_wanted_skills_by_country(self, country: str, skill_type: str = 'languages', n: int = 100) -> pd.Series:
        """Get wanted skills ONLY for specific country"""
        country_df = self.df[self.df['country'] == country]
        
        if country_df.empty:
            return pd.Series()
        
        col_map = {
            'languages': 'wanted_languages',
            'frameworks': 'wanted_frameworks',
            'databases': 'wanted_databases',
            'platforms': 'wanted_platforms'
        }
        
        col = col_map.get(skill_type, 'wanted_languages')
        
        return (country_df[col]
                .dropna()
                .str.split(';')
                .explode()
                .str.strip()
                .str.title()
                .pipe(lambda x: x[x != ''])
                .value_counts()
                .head(n))

    def get_skill_gap(self, skill_type: str = 'languages', n: int = 10) -> pd.DataFrame:
        """
        Identify skill gaps (wanted but not commonly had)
        """
        have_skills = self.get_top_skills(skill_type, n=50)
        want_skills = self.get_wanted_skills(skill_type, n=50)

        gaps = []
        for skill in want_skills.index:
            have = have_skills.get(skill, 0)
            want = want_skills[skill]
            gap = want - have
            if gap > 0:
                gaps.append({
                    'skill': skill,
                    'have_count': have,
                    'want_count': want,
                    'gap': gap
                })

        gaps_df = pd.DataFrame(gaps)
        return gaps_df.sort_values('gap', ascending=False).head(n)

    def get_skills_by_country(self, country: str, skill_type: str = 'languages', n: int = 5) -> pd.Series:
        """Get top skills for a specific country"""
        country_data = self.df[self.df['country'] == country]
        col_map = {
            'languages': 'skills_languages',
            'frameworks': 'skills_frameworks',
            'databases': 'skills_databases',
            'platforms': 'skills_platforms'
        }

        return (country_data[col_map[skill_type]]
                .dropna()
                .str.split(';')
                .explode()
                .str.strip()
                .str.title()
                .pipe(lambda x: x[x != ''])
                .value_counts()
                .head(n))

    # ==================== JOB TITLE ANALYSIS ====================

    def get_top_job_titles(self, n: int = 5, exclude_unspecified: bool = True) -> pd.Series:
        """Get top N job titles (default 5)"""
        titles = self.df['job_title'].value_counts()
        if exclude_unspecified:
            titles = titles[titles.index != 'Not specified']
        return titles.head(n)

    # ==================== TREND ANALYSIS ====================

    def get_emerging_skills(self, skill_type: str = 'languages', n: int = 10) -> pd.DataFrame:
        """
        Identify emerging skills based on high want-to-have ratio
        """
        have_skills = self.get_top_skills(skill_type, n=100)
        want_skills = self.get_wanted_skills(skill_type, n=100)

        emerging = []
        for skill in want_skills.index:
            have = have_skills.get(skill, 1)
            want = want_skills[skill]
            ratio = want / have

            if ratio > 0.5 and have > 10:
                emerging.append({
                    'skill': skill,
                    'want_count': want,
                    'have_count': have,
                    'demand_ratio': round(ratio, 2)
                })

        emerging_df = pd.DataFrame(emerging)
        return emerging_df.sort_values('demand_ratio', ascending=False).head(n)

    # ==================== SKILL RECOMMENDATIONS ====================

    def recommend_skills(self, user_skills: List[str],
                         career_goal: Optional[str] = None,
                         country: Optional[str] = None,
                         n: int = 5) -> pd.DataFrame:
        """
        Recommend skills to learn based on user's current skills and career goals
        """
        recommendations = []

        # Get all potential skills to recommend
        all_skills = self.get_top_skills('languages', n=100).index.tolist()
        all_skills = list(set(all_skills))

        # Remove skills user already has
        user_skills_clean = [s.strip().title() for s in user_skills]
        candidate_skills = [s for s in all_skills if s not in user_skills_clean]

        # ========== NORMALISASI JOB TITLE ==========
        if career_goal:
            career_lower = career_goal.lower()
            if 'data scientist' in career_lower:
                career_goal = 'Data Scientist'
            elif 'web developer' in career_lower:
                career_goal = 'Web Developer'
            elif 'mobile developer' in career_lower:
                career_goal = 'Mobile Developer'
            elif 'devops engineer' in career_lower:
                career_goal = 'DevOps Engineer'
            elif 'machine learning engineer' in career_lower:
                career_goal = 'Machine Learning Engineer'
        # ==========================================

        # ========== WHITELIST FILTER ==========
        if career_goal and career_goal in self.RELEVANT_SKILLS_BY_JOB:
            allowed_skills = self.RELEVANT_SKILLS_BY_JOB[career_goal]
            candidate_skills = [s for s in candidate_skills if s in allowed_skills]
        # =====================================

        if not candidate_skills:
            return pd.DataFrame()

        # ========== GET DEMAND DATA (GLOBAL OR COUNTRY-SPECIFIC) ==========
        if country:
            wanted = self.get_wanted_skills_by_country(country, 'languages', 100)
        else:
            wanted = self.get_wanted_skills('languages', 100)
        # =================================================================

        for skill in candidate_skills:
            want_count = wanted.get(skill, 0)
            reasons = []

            if want_count > 0:
                reasons.append(f"{want_count} developers want to learn this")

            recommendations.append({
                'skill': skill,
                'want_count': want_count,
                'reasons': ', '.join(reasons)
            })

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('want_count', ascending=False).head(n)

        return rec_df

    def get_recommendations_text(self, user_skills: List[str],
                                 career_goal: Optional[str] = None,
                                 country: Optional[str] = None,
                                 n: int = 3) -> str:
        """Return text for skill recommendations (for dashboard)"""

        recs = self.recommend_skills(user_skills, career_goal, country, n)

        if recs.empty:
            return "⚠️ No recommendations available\n"

        text = ""
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            text += f"{i}. **{row['skill']}**\n"
            text += f"   → {row['reasons']}\n\n"

        if not recs.empty:
            top_skill = recs.iloc[0]['skill']
            text += f"💡 **Top Priority:** Start with {top_skill} — it's the most in-demand skill!\n"

        return text

    # ==================== HIRING PRIORITIES ====================

    def get_hiring_priorities(self, country: Optional[str] = None) -> pd.Series:
        """Analyze what recruiters prioritize in hiring"""
        hiring_cols = ['hiring_communication', 'hiring_tech_exp',
                       'hiring_algorithms', 'hiring_education',
                       'hiring_opensource', 'hiring_execution']

        data = self.df
        if country:
            data = data[data['country'] == country]

        name_map = {
            'communication': 'Communication',
            'execution': 'Getting Things Done',
            'algorithms': 'Problem Solving',
            'tech_exp': 'Hands-on Skills',
            'opensource': 'Open Source Work',
            'education': 'College Degree'
        }

        priorities = {}
        for col in hiring_cols:
            if col in data.columns:
                important_pct = ((data[col] == 'Important') |
                                 (data[col] == 'Very important')).mean() * 100
                key = col.replace('hiring_', '')
                priorities[name_map.get(key, key)] = important_pct

        return pd.Series(priorities).sort_values(ascending=False)

    # ==================== TEXT OUTPUT METHODS ====================

    def get_top_skills_text(self, skill_type: str = 'languages', n: int = 3) -> str:
        """Return simple text for top skills"""
        skills = self.get_top_skills(skill_type, n)

        text = ""
        for i, (skill, count) in enumerate(skills.items(), 1):
            text += f"{i}. **{skill}** — {count:,} developers\n"

        return text

    def get_wanted_skills_text(self, skill_type: str = 'languages', n: int = 3) -> str:
        """Return simple text for most wanted skills"""
        skills = self.get_wanted_skills(skill_type, n)

        text = ""
        for i, (skill, count) in enumerate(skills.items(), 1):
            text += f"{i}. **{skill}** — wanted by {count:,} developers\n"

        return text

    def get_skill_gaps_text(self, skill_type: str = 'languages', n: int = 3) -> str:
        """Return simple text for skill gaps"""
        gaps = self.get_skill_gap(skill_type, n)

        if gaps.empty:
            return "⚠️ No skill gaps found\n"

        text = ""
        for _, row in gaps.iterrows():
            text += f"• **{row['skill']}** — {row['gap']:,} more want it than have it\n"

        top_skill = gaps.iloc[0]['skill']
        text += f"\n💡 **Insight:** Learn {top_skill} — high demand, low supply!\n"

        return text

    def get_emerging_skills_text(self, skill_type: str = 'languages', n: int = 3) -> str:
        """Return simple text for emerging skills"""
        emerging = self.get_emerging_skills(skill_type, n)

        if emerging.empty:
            return "⚠️ No emerging skills identified\n"

        text = ""
        for _, row in emerging.iterrows():
            text += f"• **{row['skill']}** — {row['demand_ratio']}x demand growth\n"

        text += f"\n💡 **Insight:** These skills are growing fastest! Learn them early.\n"

        return text

    def get_hiring_priorities_text(self, country: Optional[str] = None, n: int = 3) -> str:
        """Return simple text for hiring priorities"""
        priorities = self.get_hiring_priorities(country)

        if priorities.empty:
            return "⚠️ No hiring priority data available\n"

        text = ""
        for name, pct in priorities.head(n).items():
            text += f"• **{name}** — {pct:.0f}% of recruiters\n"

        if not priorities.empty:
            text += f"\n💡 **Key Insight:** {priorities.index[0]} is what recruiters value most!\n"

        return text

    def get_top_job_titles_text(self, n: int = 3) -> str:
        """Return simple text for top job titles"""
        titles = self.get_top_job_titles(n)

        text = ""
        for i, (title, count) in enumerate(titles.items(), 1):
            text += f"{i}. **{title}** — {count:,} developers\n"

        return text

    def get_hiring_glossary(self) -> str:
        """Return glossary explaining hiring priority terms"""
        text = """
| Term | What It Means |
|------|---------------|
| **Communication** | Can you explain ideas clearly? Write well? Work with teams? |
| **Getting Things Done** | Do you finish what you start? Meet deadlines? Reliable? |
| **Problem Solving** | Can you think logically? Solve complex problems? |
| **Hands-on Skills** | Do you have actual coding experience? Built real projects? |
| **Open Source Work** | Have you contributed to public projects on GitHub? |
| **College Degree** | Do you have a formal university education? |
"""
        return text

    def get_skills_glossary(self) -> str:
        """Return glossary explaining skill-related terms"""
        text = """
| Term | What It Means |
|------|---------------|
| **Top Skills** | Most commonly used programming languages/tools |
| **Wanted Skills** | Skills developers want to learn next |
| **Skill Gap** | Skills with high demand but low supply (opportunity!) |
| **Emerging Skills** | Fastest growing skills (learn these NOW!) |
"""
        return text

    def get_dashboard_text(self) -> str:
        """Complete dashboard as text"""
        text = ""

        # Quick stats
        text += f"📈 **Total Developers:** {len(self.df):,}\n"
        text += f"🌍 **Countries:** {self.df['country'].nunique()}\n"
        text += f"💻 **Unique Skills:** {self.df['skills_languages'].str.split(';').explode().nunique()}\n\n"

        # Top skills (top 3)
        text += self.get_top_skills_text('languages', 3)
        text += "\n"

        # Skill gaps (top 3)
        text += self.get_skill_gaps_text('languages', 3)
        text += "\n"

        # Hiring priorities (top 3)
        text += self.get_hiring_priorities_text(None, 3)
        text += "\n"

        # Emerging skills (top 3)
        text += self.get_emerging_skills_text('languages', 3)

        return text


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize the platform
    jmi = JobMarketIntelligence("cleaned_data.csv")

    # Display full dashboard
    print(jmi.get_dashboard_text())

    print("\n" + "=" * 55)
    print("INDIVIDUAL SECTIONS (Top 3)")
    print("=" * 55)

    print(jmi.get_top_skills_text('languages', 3))
    print(jmi.get_skill_gaps_text('languages', 3))
    print(jmi.get_hiring_priorities_text('United States', 3))
    print(jmi.get_emerging_skills_text('languages', 3))
    print(jmi.get_top_job_titles_text(3))

    # Print glossaries
    print("\n" + "=" * 55)
    print("GLOSSARIES")
    print("=" * 55)
    print(jmi.get_hiring_glossary())
    print(jmi.get_skills_glossary())

    # Example: Get skill recommendations with DIFFERENT COUNTRIES
    print("\n" + "=" * 55)
    print("SKILL RECOMMENDATIONS - INDONESIA")
    print("=" * 55 + "\n")

    my_skills = ["Javascript", "Python"]
    career_goal = "Web Developer"
    print(jmi.get_recommendations_text(my_skills, career_goal, country="Indonesia", n=3))
    
    print("\n" + "=" * 55)
    print("SKILL RECOMMENDATIONS - CHINA")
    print("=" * 55 + "\n")
    
    print(jmi.get_recommendations_text(my_skills, career_goal, country="China", n=3))
    
    print("\n" + "=" * 55)
    print("SKILL RECOMMENDATIONS - UNITED STATES")
    print("=" * 55 + "\n")
    
    print(jmi.get_recommendations_text(my_skills, career_goal, country="United States", n=3))