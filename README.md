# 🎯 Skill Recommendation App

A Python-based application that recommends skills based on user input,
with country filtering and whitelist support.

<img width="600" alt="Home Page" src="https://github.com/user-attachments/assets/61e33fb7-4bb4-4232-93be-ee8195f3d194" />

<img width="600" alt="Results Page" src="https://github.com/user-attachments/assets/13269785-4328-4287-85ec-452f4afe141e" />

<img width="600" alt="Filter Page" src="https://github.com/user-attachments/assets/6178b77e-dd5f-4861-8ebf-c6c1c533e6bd" />

## ✨ Features
- Skill recommendations with whitelist filtering
- Country-based filtering
- Data cleaning + exploratory data analysis (EDA)
- Interactive frontend

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data:** Pandas, NumPy
- **Notebook:** Jupyter
- **Frontend:** Streamlit 

## 📊 Dataset

This project uses the **Stack Overflow Developer Survey** dataset,
which captures developer demographics, skills, salaries, and technology usage
from thousands of respondents worldwide.

- **Source:** [Stack Overflow Developer Survey](https://openi.pcl.ac.cn/AIisAllweNeed/Stack-Overflow-Developer-Survey/datasets)
- **Hosted on:** OpenI (Peng Cheng Laboratory)
- **Original survey:** [Stack Overflow Annual Developer Survey](https://survey.stackoverflow.co/)
- **Format:** CSV
- **Processing:** Cleaned and preprocessed in `notebooks/01_eda.ipynb`

### Why This Dataset?
We explored real-time / live APIs for developer skills data, but most options:
- Had strict API rate limits
- Required paid access
- Lacked tech-specific data

The Stack Overflow Developer Survey provided the most comprehensive,
reliable, and tech-focused dataset for our recommendation engine.
## 📁 Project Structure
```
├── frontend.py            # Main app entry point
├── trial_new.py           # Skill recommendation logic
├── EDA.ipynb              # Exploratory data analysis
├── cleaned_data.csv       # Processed dataset
├── requirements.txt       # Dependencies
└── README.md
```

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/aryd11/frontend.git
cd frontend
pip install -r requirements.txt
```

### Run
```bash
streamlit run frontend.py
# or
python frontend.py
```

## 👥 Collaborators
Built in collaboration by:
- **Elvaretta Nafisah** — [@elvarettanafisah-cloud](https://github.com/elvarettanafisah-cloud)
- **Tantiana Indira** — [@aryd11](https://github.com/aryd11)
- **Leonard Mutungi**

## 📄 License
MIT License
