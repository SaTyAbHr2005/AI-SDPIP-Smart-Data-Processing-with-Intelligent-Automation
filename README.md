# 📊 AI-Enhanced Survey Analysis

> AI-powered survey data cleaning, analysis, and estimation for government statistics modernization.

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
```

Open your browser at http://localhost:8501.

---

## 📖 Usage

| Step | Action                     | Output                            |
| ---- | -------------------------- | --------------------------------- |
| 1    | 📁 Upload Data (CSV/Excel) | Loads dataset for processing      |
| 2    | 🤖 AI Processing           | Cleans and imputes missing values |
| 3    | 📊 Generate Estimates      | Choose estimation method          |
| 4    | 📄 Export Reports          | Download data and reports         |

---

## 📂 Project Structure

```plaintext
mospi-ai-survey-analysis/
├── app.py               # Streamlit app entry point
├── config.py            # Configuration settings
├── data_processor.py    # Data cleaning functions
├── ai_text_processor.py # AI text processing
├── hybrid_processor.py  # Hybrid pipeline logic
├── ai_estimator.py      # AI estimation methods
├── report_generator.py  # Report generation
├── utils.py             # Helper utilities
└── requirements.txt     # Dependencies list
```

---

## 📋 Requirements

streamlit==1.28.0
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
plotly==5.15.0
scipy==1.11.1
statsmodels==0.14.0
openpyxl==3.1.2

---

## 🤝 Contributing

# Fork the repository

# Create a feature branch
git checkout -b feature/my-feature

# Commit your changes
git commit -m "Add new feature"

# Push the branch
git push origin feature/my-feature

Then open a Pull Request.

---

## 🏆 Acknowledgments

- MoSPI – For initiating the modernization challenge

- Open-Source Community – For the tools and libraries

---

⭐ If this project helps you, star the repository to support development.
