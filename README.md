# Bank Campaign Analytics and Term Deposit Response Prediction Using Machine Learning


- **Student Name:** Mohan Kumar M
- **Project Title:** Bank Campaign Analytics and Term Deposit Response Prediction Using Machine Learning
- **Target Variable:** `y` (`yes` = customer subscribed to a term deposit; `no` = customer did not subscribe)
- **Dataset Source:** UCI Machine Learning Repository — Bank Marketing Dataset (`bank.csv`)

---

## 1. Project Overview

Commercial retail banks frequently conduct direct telemarketing campaigns to persuade customers to open long-term deposit accounts. Term deposits provide financial institutions with a stable, predictable base of low-cost capital to underwrite mortgages and loans. However, cold outbound telemarketing campaigns typically experience low baseline conversion rates (8% to 12%), resulting in substantial call-center expenses, staff fatigue, and customer dissatisfaction.

This project delivers an end-to-end data analytics and applied machine learning solution designed for a fourth-year engineering academic standard. It analyzes real-world historical campaign interactions, formalizes executive Key Performance Indicators (KPIs), engineers domain-specific features, and develops production-grade classification pipelines comparing **Logistic Regression** and **Random Forest**. Crucially, the project demonstrates and resolves the phenomenon of **data leakage** caused by post-contact call duration.

---

## 2. Business Problem & Central Research Question

> *"Which customer and campaign characteristics are associated with successful term-deposit subscriptions, and can machine learning predict whether a customer is likely to subscribe before making contact?"*

By predicting customer propensity *prior to dialing*, the bank can prioritize high-probability prospects, optimize call-center allocation, and reduce customer annoyance caused by unproductive cold calls.

---

## 3. Project Objectives

1. **Data Auditing & Validation:** Verify the structural completeness and authenticity of the UCI Bank Marketing benchmark dataset.
2. **Exploratory Data Analysis:** Uncover demographic, financial, and behavioral patterns associated with deposit uptake without drawing unfounded causal claims.
3. **Executive KPI Formalization:** Measure baseline campaign conversion rates, customer wealth dispersion, and historical re-engagement yield.
4. **Domain Feature Engineering:** Transform continuous and sentinel variables into meaningful business segments (`age_group`, `balance_group`, `previously_contacted`, `contact_frequency`).
5. **Data Leakage Resolution:** Empirically demonstrate how post-contact variables (call duration) falsely inflate evaluation metrics, and enforce strict pre-contact model design.
6. **Predictive Modeling & Trade-off Evaluation:** Train and evaluate balanced Logistic Regression and Random Forest classifiers, assessing the trade-off between lead volume (Recall) and operational efficiency (Precision).
7. **Actionable Recommendations:** Provide evidence-based operational guidelines for bank campaign planning and lead prioritization.

---

## 4. Dataset Description & Official Citation

The dataset utilized in this project is the authentic **Bank Marketing dataset** from the **UCI Machine Learning Repository**. The project specifically uses the benchmark `bank.csv` dataset (4,521 customer records and 17 attributes).

- **Official UCI Dataset URL:** [https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing)
- **Direct Archive Link:** `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
- **Usable Local File Path:** `data/bank.csv`

### Official Academic Citation
```bibtex
@article{moro2014data,
  title={A data-driven approach to predict the success of bank telemarketing},
  author={Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo},
  journal={Decision Support Systems},
  volume={62},
  pages={22--31},
  year={2014},
  publisher={Elsevier},
  doi={10.1016/j.dss.2014.03.001}
}
```

---

## 5. Project Directory Structure

```text
Bank-Campaign-Analytics/
│
├── MohanKumarM_BankCampaignAnalytics.ipynb       # Mandatory submission: fully executed 17-section notebook
├── MohanKumarM_ProjectReport.docx               # Mandatory submission: comprehensive 24-chapter report with dashboard screenshots
├── requirements.txt                              # Mandatory submission: pinned python dependencies
├── README.md                                     # Mandatory submission: complete project documentation
├── app.py                                        # Optional interactive Streamlit web dashboard
├── .gitignore                                    # Git ignore rules for virtualenvs and checkpoints
│
├── data/
│   └── bank.csv                                  # Authentic UCI Bank Marketing dataset (4,521 rows, 17 cols)
│
├── outputs/
│   ├── figures/                                  # 14 high-resolution (300 DPI) visualization figures
│   │   ├── eda_target_distribution.png
│   │   ├── eda_job_subscription.png
│   │   ├── eda_education_subscription.png
│   │   ├── eda_housing_loan.png
│   │   ├── eda_balance_distribution.png
│   │   ├── eda_campaign_contacts.png
│   │   ├── eda_poutcome_subscription.png
│   │   ├── eda_duration_analysis.png
│   │   ├── leakage_comparison.png
│   │   ├── model_cm_logistic_regression.png
│   │   ├── model_cm_random_forest.png
│   │   ├── model_roc_curves.png
│   │   ├── model_rf_feature_importance.png
│   │   └── model_lr_coefficients.png
│   │
│   └── model_results/                            # Empirical CSV evaluation metrics and summary tables
│       ├── business_kpis.csv
│       ├── leakage_experiment_results.csv
│       ├── model_comparison_metrics.csv
│       ├── rf_feature_importance.csv
│       └── lr_coefficients.csv
│
└── scripts/                                      # Automation scripts
    ├── generate_analytics_and_figures.py
    ├── build_notebook.py
    └── build_report.py
```

---

## 6. Technologies & Environment

- **Programming Language:** Python 3.10+ (Tested on Python 3.13)
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn`
- **Report Generation:** `python-docx`
- **Interactive Computing:** `jupyter`, `ipykernel`, `nbclient`
- **Optional Web Dashboard:** `streamlit`

---

## 7. Step-by-Step Installation & Execution Guide

### Step 1: Clone or Navigate to the Project Directory
Open your command terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:
```bash
cd MohanKumarM_BankCampaignAnalytics
```

### Step 2: Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
Install the required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
Launch Jupyter Notebook to inspect and run the analytical workflow:
```bash
jupyter notebook
```
In your web browser interface, open:
`MohanKumarM_BankCampaignAnalytics.ipynb`

To run all cells sequentially from the Jupyter interface:
- Click **Kernel** → **Restart & Run All Cells**.

### Step 5 (Optional): Run the Interactive Streamlit Dashboard
Launch the interactive web application to explore KPIs and test pre-contact lead predictions:
```bash
streamlit run app.py
```

---

## 8. Critical Finding: The Data Leakage Pitfall

In telemarketing datasets, the variable `duration` records the length of the call in seconds. 

### Why `duration` Causes Data Leakage
- **Pre-Contact Reality:** Before a customer is called, the call length is identically $0$ seconds. 
- **Post-Contact Reality:** If a customer is interested, the call naturally extends to discuss terms, verify personal details, and finalize the agreement. Thus, long call duration is a *consequence* of subscription, not a cause.
- **The Danger:** A model trained with `duration` achieves an artificially inflated ROC-AUC of **0.8959**. However, this model is completely non-functional when deployed in production dialers because future call duration cannot be known prior to placing the call.

### Empirical Demonstration (Holdout Test Set)

| Model Configuration | Operational Feasibility | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **WITH Duration (Data Leakage)** | ❌ **NOT FEASIBLE** (Requires future talk time) | 88.73% | 54.55% | 11.54% | 0.1905 | **0.8959** |
| **WITHOUT Duration (Clean Pre-Contact)** | ✅ **FEASIBLE** (Known prior to call) | 88.73% | 55.56% | 9.62% | 0.1639 | **0.7185** |

> **Methodological Decision:** In this project, `duration` is analyzed strictly for descriptive observation during EDA and is **strictly excluded** from all pre-contact prediction models.

---

## 9. Machine Learning Evaluation & Model Comparison

Both pre-contact models were trained using an 80/20 stratified train-test split (`random_state=42`, `stratify=y`) on features available before contact. Preprocessing (StandardScaler for numerics, OneHotEncoder for categoricals) was managed using `ColumnTransformer` inside scikit-learn `Pipeline` objects.

### Empirical Performance Comparison on Holdout Test Set (905 Records)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 72.49% | 23.64% | **62.50%** | **0.3430** | **0.7470** |
| **Random Forest (Balanced, 200 Trees)** | **88.73%** | **55.56%** | 9.62% | 0.1639 | 0.7185 |

### Operational Trade-off Analysis
- **Choose Logistic Regression when:** The commercial strategy emphasizes **customer acquisition volume**. With a Recall of **62.5%**, the model captures nearly two-thirds of all potential subscribers in the database, doubling the random dialing conversion rate (23.6% Precision vs 11.5% baseline).
- **Choose Random Forest when:** Call-center capacity and budget are **strictly constrained**. With a Precision of **55.6%**, more than half of all prioritized calls result in a deposit subscription, minimizing wasted advisor talk time.

---

## 10. Key Analytical Findings & Business Recommendations

1. **Prior Campaign Success is the Strongest Predictor:** Customers with a successful prior campaign outcome (`poutcome == 'success'`) converted at **64.3%** (vs 11.5% baseline).  
   *Recommendation:* Implement an automated re-engagement queue that contacts previously converted clients first.
2. **Debt Commitments Constrain Investment:** Customers with housing mortgages converted at **8.6%** (vs 15.3% for debt-free clients), while personal debt halved conversion to **6.7%**.  
   *Recommendation:* Suppress term deposit pitches for clients servicing significant debt; offer liquid savings or debt consolidation instead.
3. **Contact Fatigue Emerges Beyond 2 Calls:** Over 80% of conversions occur within the first 2 contact attempts. Response rates drop below 6% beyond 3 calls.  
   *Recommendation:* Institute a strict policy cap of 3 call attempts per customer per campaign to reduce call-center operating costs.
4. **Demographic Targets:** Students (22.6%) and retirees (23.5%) exhibited the highest observed response rates.  
   *Recommendation:* Develop tailored, safe, fixed-return deposit marketing packages specifically for retirement capital preservation and student savings.

---

## 11. Viva-Voce Preparation Guide

After every major notebook section, a structured **Quick Understanding** box is included. Below are the key concepts required for viva defense:

1. **Q: What is the central business problem we are solving?**  
   *A: High outbound telemarketing costs and low conversion rates (~11.5%). We build an algorithmic pre-contact scoring model to rank and prioritize receptive leads before agents dial.*
2. **Q: Why is `duration` excluded from the final machine learning pipeline?**  
   *A: Because call duration is a post-contact variable. Including it creates data leakage, yielding an artificially high ROC-AUC (0.896) that cannot be used in practice because call length is unknown before dialing.*
3. **Q: Why is raw accuracy a misleading metric for this dataset?**  
   *A: Because the dataset has an 8:1 class imbalance (88.5% non-subscribers). A naive model predicting 'no' for everyone achieves 88.5% accuracy while finding zero subscribers. ROC-AUC, Precision, and Recall are much more informative.*
4. **Q: How does `ColumnTransformer` prevent data leakage during preprocessing?**  
   *A: It fits statistical transformers (such as scaler means and one-hot encoder categories) exclusively on the training partition, ensuring test data remains completely unseen.*
5. **Q: Which model should the bank deploy in production?**  
   *A: It depends on operational resources. If call center hours are abundant, use Logistic Regression (Recall = 62.5% to maximize conversions). If call center capacity is scarce, use Random Forest (Precision = 55.6% to maximize advisor efficiency).*

---

## 12. Project Deliverables Checklist

- [x] Authentic UCI dataset downloaded to `data/bank.csv` (4,521 rows, 17 columns)
- [x] Fully executed Jupyter Notebook: `MohanKumarM_BankCampaignAnalytics.ipynb` (17 sections, 63 cells, 0 errors)
- [x] Academic Project Report: `MohanKumarM_ProjectReport.docx` (24 chapters, 17 embedded figures including real dashboard screenshots, formatted tables)
- [x] Pinned dependencies file: `requirements.txt`
- [x] Complete documentation: `README.md`
- [x] Version control rules: `.gitignore`
- [x] Interactive dashboard: `app.py`
- [x] Clean, reproducible empirical figures in `outputs/figures/`
- [x] Summary metrics in `outputs/model_results/`

---

## 13. Project Metadata & Academic Certification

- **Student Name:** Mohan Kumar M
- **Degree:** Bachelor of Engineering / Technology (Fourth Year)
- **Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
- **Partner Organization:** BharatCares in association with All India Council for Technical Education (AICTE)
- **Academic Year:** 2026
