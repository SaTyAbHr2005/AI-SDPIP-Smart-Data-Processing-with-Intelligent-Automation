# 📊 MoSPI AI-Enhanced Survey Analysis

> AI-powered survey data cleaning, analysis, and estimation for government statistics modernization.

This project is built for the **Ministry of Statistics and Programme Implementation (MoSPI)** as part of **Statathon 2025**.  
It integrates **traditional statistical methods** with **AI/ML models** to automate survey data processing, impute missing values, generate estimates, and produce professional reports.

---

## 🚀 Features

- 🤖 **AI Hybrid Processing** – ML-based and rule-based cleaning with intelligent imputation
- 📈 **Multiple Estimation Methods** – Design-based, model-based, and hybrid estimation
- 📊 **Interactive Dashboards** – Real-time visualizations via Plotly
- 📄 **Report Generation** – Export ready-to-use reports in HTML, PDF-ready, and JSON formats
- 🔍 **Data Quality Scoring** – Completeness, reliability, and consistency metrics

---

## 🛠 Tech Stack

| Category      | Technologies |
|---------------|--------------|
| Frontend      | Streamlit, Plotly |
| Backend       | Python 3.8+, Pandas, NumPy |
| ML/AI         | scikit-learn (Random Forest, Gradient Boosting, Isolation Forest) |
| Statistics    | SciPy, statsmodels |
| File Formats  | CSV, Excel (.xlsx, .xls) |

---

## 📦 Installation

### **Prerequisites**
- Python 3.8 or higher
- 4 GB+ RAM recommended

### **Setup**

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/mospi-ai-survey-analysis.git
cd mospi-ai-survey-analysis

# 2️⃣ Create a virtual environment
python -m venv venv

# 3️⃣ Activate the virtual environment
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# 4️⃣ Install dependencies
pip install -r requirements.txt

# 5️⃣ Run the application
streamlit run app.py
