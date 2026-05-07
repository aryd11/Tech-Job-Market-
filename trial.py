import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class JobMarketIntelligence:
    """
    Job Market Intelligence Platform
    Analyzes job market data to identify trends, in-demand skills, and provide recommendations
    """

    def __init__(self, data_path: str = "cleaned_data.csv"):
        """
        Initialize the platform with survey data

        Parameters:
        -----------
        data_path : str
            Path to the Stack Overflow survey data CSV file
        """
        print("🚀 Initializing Job Market Intelligence Platform...")
        self.df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.df):,} developer responses")

    # ==================== SKILL ANALYSIS METHODS ====================

    def get_top_skills(self, skill_type: str = 'languages', n: int = 10) -> pd.Series:
        """
        Get top N most common skills

        Parameters:
        -----------
        skill_type : str
            'languages', 'frameworks', 'databases', or 'platforms'
        n : int
            Number of top skills to return
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

    def get_skill_gap(self, skill_type: str = 'languages', n: int = 10) -> pd.DataFrame:
        """
        Identify skill gaps (wanted but not commonly had)

        Returns:
        --------
        DataFrame with columns: skill, have_count, want_count, gap
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

    def get_top_job_titles(self, n: int = 10, exclude_unspecified: bool = True) -> pd.Series:
        """Get top N job titles"""
        titles = self.df['job_title'].value_counts()
        if exclude_unspecified:
            titles = titles[titles.index != 'Not specified']
        return titles.head(n)

    def extract_job_categories(self) -> pd.Series:
        """Extract broad job categories from job titles"""
        categories = {
            'Web Developer': ['web', 'frontend', 'backend', 'full stack'],
            'Mobile Developer': ['mobile', 'ios', 'android'],
            'Data Scientist': ['data scientist', 'machine learning', 'ml', 'ai'],
            'DevOps Engineer': ['devops', 'sre', 'infrastructure'],
            'Desktop Developer': ['desktop', 'windows', 'macos'],
            'Embedded Developer': ['embedded', 'firmware', 'iot'],
            'Database Administrator': ['database', 'dba', 'data engineer'],
            'Systems Administrator': ['system admin', 'sysadmin', 'it'],
            'Security Engineer': ['security', 'cybersecurity'],
            'Game Developer': ['game', 'graphics']
        }

        def categorize(title):
            if pd.isna(title) or title == 'Not specified':
                return 'Not specified'
            title_lower = str(title).lower()
            for category, keywords in categories.items():
                if any(kw in title_lower for kw in keywords):
                    return category
            return 'Other'

        return self.df['job_title'].apply(categorize)

    # ==================== TREND ANALYSIS ====================

    def get_emerging_skills(self, skill_type: str = 'languages', n: int = 10) -> pd.DataFrame:
        """
        Identify emerging skills based on high want-to-have ratio

        Returns:
        --------
        DataFrame with columns: skill, want_count, have_count, demand_ratio
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

    # ==================== HIRING PRIORITIES ====================

    def get_hiring_priorities(self, country: Optional[str] = None) -> pd.Series:
        """Analyze what employers prioritize in hiring"""
        hiring_cols = ['hiring_communication', 'hiring_tech_exp',
                       'hiring_algorithms', 'hiring_education',
                       'hiring_opensource', 'hiring_execution']

        data = self.df
        if country:
            data = data[data['country'] == country]

        priorities = {}
        for col in hiring_cols:
            if col in data.columns:
                important_pct = ((data[col] == 'Important') |
                                 (data[col] == 'Very important')).mean() * 100
                priorities[col.replace('hiring_', '')] = important_pct

        return pd.Series(priorities).sort_values(ascending=False)

    # ==================== SKILL RECOMMENDATIONS ====================

    def recommend_skills(self, user_skills: List[str],
                         career_goal: Optional[str] = None,
                         country: Optional[str] = None,
                         n: int = 5) -> pd.DataFrame:
        """
        Recommend skills to learn based on user's current skills and career goals

        Parameters:
        -----------
        user_skills : list
            List of skills the user currently has
        career_goal : str, optional
            Target job title or role
        country : str, optional
            Target country for job market
        n : int
            Number of recommendations to return

        Returns:
        --------
        DataFrame with columns: skill, demand_score, recommendation_reason
        """
        recommendations = []

        all_skills = self.get_top_skills('languages', n=50).index.tolist()
        all_skills.extend(self.get_top_skills('frameworks', n=30).index.tolist())
        all_skills = list(set(all_skills))

        user_skills_clean = [s.strip().title() for s in user_skills]
        candidate_skills = [s for s in all_skills if s not in user_skills_clean]

        wanted = self.get_wanted_skills('languages', n=100)
        wanted_frameworks = self.get_wanted_skills('frameworks', n=100)

        for skill in candidate_skills:
            score = 0
            reasons = []

            want_count = wanted.get(skill, 0) + wanted_frameworks.get(skill, 0)
            if want_count > 0:
                score += min(want_count / 500, 5)
                reasons.append(f"Wanted by {want_count} developers")

            have_count = self.get_top_skills('languages', n=100).get(skill, 0)
            have_count += self.get_top_skills('frameworks', n=100).get(skill, 0)

            if have_count > 0:
                gap_ratio = want_count / have_count if have_count > 0 else 0
                score += min(gap_ratio * 2, 3)
                if gap_ratio > 1:
                    reasons.append(f"High demand vs availability ({gap_ratio:.1f}x)")

            if career_goal and self._is_skill_relevant_for_job(skill, career_goal):
                score += 2
                reasons.append(f"Relevant for {career_goal} roles")

            if country:
                country_skills = self.get_skills_by_country(country, 'languages', n=20)
                if skill in country_skills.index:
                    score += 1.5
                    reasons.append(f"Popular in {country}")

            recommendations.append({
                'skill': skill,
                'demand_score': round(score, 1),
                'reasons': ', '.join(reasons[:3])
            })

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('demand_score', ascending=False).head(n)

        return rec_df

    def _is_skill_relevant_for_job(self, skill: str, job_title: str) -> bool:
        """Check if a skill is relevant for a specific job title"""
        job_lower = job_title.lower()
        skill_lower = skill.lower()

        job_skill_map = {
            'web': ['javascript', 'react', 'angular', 'vue', 'html', 'css', 'node', 'django', 'flask'],
            'data': ['python', 'r', 'sql', 'pandas', 'tensorflow', 'pytorch', 'scikit'],
            'mobile': ['swift', 'kotlin', 'react native', 'flutter', 'android', 'ios'],
            'devops': ['docker', 'kubernetes', 'aws', 'azure', 'terraform', 'jenkins'],
            'backend': ['java', 'python', 'go', 'c#', 'spring', 'django', 'node'],
            'frontend': ['javascript', 'react', 'angular', 'vue', 'typescript', 'html', 'css'],
            'game': ['c++', 'unity', 'unreal', 'c#'],
            'embedded': ['c', 'c++', 'rust', 'assembly', 'arduino']
        }

        for key, skills in job_skill_map.items():
            if key in job_lower and any(s in skill_lower for s in skills):
                return True
        return False

    # ==================== VISUALIZATION METHODS ====================

    def plot_top_skills(self, skill_type: str = 'languages', n: int = 10, save_path: Optional[str] = None):
        """Create bar chart of top skills"""
        skills = self.get_top_skills(skill_type, n)

        plt.figure(figsize=(10, 6))
        skills.plot(kind='barh')
        plt.xlabel('Number of Developers')
        plt.ylabel('Skill')
        plt.title(f'Top {n} {skill_type.title()} Skills')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_skill_gap(self, skill_type: str = 'languages', n: int = 10, save_path: Optional[str] = None):
        """Create bar chart of skill gaps"""
        gaps = self.get_skill_gap(skill_type, n)

        if len(gaps) == 0:
            print("No skill gaps found")
            return

        plt.figure(figsize=(12, 6))
        x = range(len(gaps))
        width = 0.35

        plt.bar([i - width / 2 for i in x], gaps['have_count'], width, label='Have', color='skyblue')
        plt.bar([i + width / 2 for i in x], gaps['want_count'], width, label='Want', color='salmon')

        plt.xlabel('Skill')
        plt.ylabel('Number of Developers')
        plt.title(f'Top {n} {skill_type.title()} Skill Gaps')
        plt.xticks(x, gaps['skill'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_emerging_skills(self, skill_type: str = 'languages', n: int = 10, save_path: Optional[str] = None):
        """Create bar chart of emerging skills based on demand ratio"""
        emerging = self.get_emerging_skills(skill_type, n)

        if len(emerging) == 0:
            print("No emerging skills identified")
            return

        plt.figure(figsize=(10, 6))
        plt.barh(emerging['skill'], emerging['demand_ratio'], color='purple')
        plt.xlabel('Demand Ratio (Want / Have)')
        plt.ylabel('Skill')
        plt.title(f'Top {n} Emerging {skill_type.title()} Skills')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_hiring_priorities(self, country: Optional[str] = None, save_path: Optional[str] = None):
        """Create bar chart of hiring priorities"""
        priorities = self.get_hiring_priorities(country)

        if priorities.empty:
            print("No hiring priority data available.")
            return

        plt.figure(figsize=(10, 6))
        priorities.plot(kind='bar')
        plt.xlabel('Priority Area')
        plt.ylabel('Percentage (%)')
        title = 'Hiring Priorities'
        if country:
            title += f' - {country}'
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    # ==================== REPORT GENERATION ====================

    def generate_market_report(self, country: Optional[str] = None) -> str:
        """Generate a comprehensive market report"""
        report = []
        report.append("=" * 70)
        report.append("📊 JOB MARKET INTELLIGENCE REPORT")
        report.append("=" * 70)

        if country:
            report.append(f"\n📍 FOCUS COUNTRY: {country.upper()}")

        report.append("\n💻 1. MOST IN-DEMAND SKILLS")
        report.append("-" * 40)
        top_skills = self.get_top_skills('languages', 5)
        for skill, count in top_skills.items():
            report.append(f"   • {skill}: {count:,} developers")

        report.append("\n🎯 2. SKILL GAPS (High Demand, Low Supply)")
        report.append("-" * 40)
        gaps = self.get_skill_gap('languages', 5)
        for _, row in gaps.iterrows():
            report.append(
                f"   • {row['skill']}: {row['want_count']:,} want, {row['have_count']:,} have (gap: {row['gap']:,})")

        report.append("\n🚀 3. EMERGING SKILLS (Fastest Growing Demand)")
        report.append("-" * 40)
        emerging = self.get_emerging_skills('languages', 5)
        for _, row in emerging.iterrows():
            report.append(f"   • {row['skill']}: {row['demand_ratio']:.1f}x demand ratio")

        report.append("\n⭐ 4. WHAT EMPLOYERS VALUE MOST")
        report.append("-" * 40)
        priorities = self.get_hiring_priorities(country)
        for priority, pct in priorities.head(3).items():
            report.append(f"   • {priority.replace('_', ' ').title()}: {pct:.0f}% of employers")

        # ✅ INI YANG DITAMBAHKAN (RETURN STATEMENT)
        return "\n".join(report)

    def generate_skill_recommendation(self, user_skills: List[str],
                                      career_goal: Optional[str] = None,
                                      country: Optional[str] = None) -> str:
        """Generate personalized skill recommendations"""
        recommendations = self.recommend_skills(user_skills, career_goal, country)

        output = []
        output.append("\n" + "=" * 60)
        output.append("🎯 PERSONALIZED SKILL RECOMMENDATIONS")
        output.append("=" * 60)
        output.append(f"\n📋 Your current skills: {', '.join(user_skills)}")

        if career_goal:
            output.append(f"🎯 Career goal: {career_goal}")
        if country:
            output.append(f"📍 Target market: {country}")

        output.append("\n💡 SKILLS TO LEARN NEXT:")
        output.append("-" * 40)

        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            output.append(f"\n{i}. {row['skill']}")
            output.append(f"   Demand Score: {row['demand_score']}/10")
            output.append(f"   Why: {row['reasons']}")

        output.append("\n" + "=" * 60)

        return "\n".join(output)