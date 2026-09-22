"""
Bank Campaign Analytics & Term Deposit Propensity Predictor
Streamlit Web Application
Student Name: Mohan Kumar M
Internship: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Configure Streamlit page
st.set_page_config(
    page_title="Bank Campaign Analytics & ML Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .leak-banner {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD DATA & CACHE MODEL
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/bank.csv', sep=';')
    df['target'] = (df['y'] == 'yes').astype(int)
    
    # Feature engineering
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 120], labels=['<30', '30-39', '40-49', '50-59', '60+'], right=False)
    def categorize_bal(b):
        if b < 0: return 'Negative'
        elif b <= 500: return 'Low (0-500)'
        elif b <= 2000: return 'Medium (501-2000)'
        else: return 'High (>2000)'
    df['balance_group'] = df['balance'].apply(categorize_bal)
    df['previously_contacted'] = np.where(df['pdays'] == -1, 'No', 'Yes')
    def categorize_freq(c):
        if c == 1: return '1 contact'
        elif c <= 3: return '2-3 contacts'
        else: return '4+ contacts'
    df['contact_frequency'] = df['campaign'].apply(categorize_freq)
    return df

@st.cache_resource
def train_precontact_model(df):
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age_group', 'balance_group', 'previously_contacted', 'contact_frequency']
    num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    
    X = df[cat_cols + num_cols]
    y = df['target']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    prep = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])
    
    pipe = Pipeline([
        ('prep', prep),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe, cat_cols, num_cols

df = load_data()
model_pipe, cat_features, num_features = train_precontact_model(df)

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/320px-IBM_logo.svg.png", width=120)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Executive Dashboard & KPIs", "🔍 Customer Segment Explorer", "🔮 Pre-Contact Propensity Predictor", "ℹ️ Academic Project Info"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Student:** Mohan Kumar M")
st.sidebar.markdown("**Program:** AICTE | IBM SkillsBuild 2026")
st.sidebar.markdown("**Partner:** BharatCares")

# -------------------------------------------------------------------
# PAGE 1: DASHBOARD & KPIS
# -------------------------------------------------------------------
if page == "📊 Executive Dashboard & KPIs":
    st.markdown('<div class="main-title">Bank Campaign Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical Key Performance Indicators from Portuguese Bank Direct Marketing Campaigns</div>', unsafe_allow_html=True)
    
    # KPI metrics row
    total_cust = len(df)
    sub_cust = int(df['target'].sum())
    conv_rate = (sub_cust / total_cust) * 100
    avg_bal = df['balance'].mean()
    med_bal = df['balance'].median()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contacts</div><div class="kpi-val">{total_cust:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Subscriptions</div><div class="kpi-val">{sub_cust:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Conversion Rate</div><div class="kpi-val">{conv_rate:.2f}%</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Mean Balance</div><div class="kpi-val">€{avg_bal:,.0f}</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Median Balance</div><div class="kpi-val">€{med_bal:,.0f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Target Distribution (Term Deposit Subscription)")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        counts = df['y'].value_counts()
        ax.bar(['No (Did Not Subscribe)', 'Yes (Subscribed)'], counts.values, color=['#3b82f6', '#10b981'], edgecolor='black', width=0.5)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 80, f"{v:,} ({v/total_cust*100:.1f}%)", ha='center', fontweight='bold')
        ax.set_ylim(0, 4700)
        ax.set_ylabel("Number of Contacts")
        plt.tight_layout()
        st.pyplot(fig)
        
    with c2:
        st.subheader("Conversion Rate by Job Category")
        fig, ax = plt.subplots(figsize=(7, 4.3))
        job_rates = df.groupby('job')['target'].mean().sort_values(ascending=True) * 100
        bars = ax.barh(job_rates.index, job_rates.values, color='#2563eb', edgecolor='black', linewidth=0.5)
        ax.axvline(conv_rate, color='#ef4444', linestyle='--', label=f'Baseline ({conv_rate:.1f}%)')
        for b in bars:
            ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, f"{b.get_width():.1f}%", va='center', fontsize=8, fontweight='bold')
        ax.set_xlim(0, 28)
        ax.set_xlabel("Subscription Rate (%)")
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Core Behavioral Drivers")
    c3, c4 = st.columns(2)
    with c3:
        st.write("**Previous Campaign Outcome (`poutcome`) vs Conversion**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        pout_rates = df.groupby('poutcome')['target'].mean().reindex(['unknown', 'failure', 'other', 'success']) * 100
        ax.bar([p.capitalize() for p in pout_rates.index], pout_rates.values, color=['#94a3b8', '#f87171', '#fbbf24', '#34d399'], edgecolor='black', width=0.5)
        ax.axhline(conv_rate, color='#ef4444', linestyle='--', label='Baseline')
        for i, v in enumerate(pout_rates.values):
            ax.text(i, v + 1.2, f"{v:.1f}%", ha='center', fontweight='bold')
        ax.set_ylim(0, 75)
        ax.set_ylabel("Subscription Rate (%)")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Clients with prior success converted at 64.3% — nearly 6x the baseline rate.")
        
    with c4:
        st.write("**Existing Loan Liabilities vs Conversion**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        hl = df.groupby('housing')['target'].mean() * 100
        pl = df.groupby('loan')['target'].mean() * 100
        labels = ['No Housing', 'Has Housing', 'No Personal', 'Has Personal']
        vals = [hl['no'], hl['yes'], pl['no'], pl['yes']]
        colors = ['#10b981', '#f59e0b', '#10b981', '#f59e0b']
        ax.bar(labels, vals, color=colors, edgecolor='black', width=0.5)
        ax.axhline(conv_rate, color='#ef4444', linestyle='--', label='Baseline')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold')
        ax.set_ylim(0, 20)
        ax.set_ylabel("Subscription Rate (%)")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Existing debt servicing significantly limits discretionary capital for term deposits.")

# -------------------------------------------------------------------
# PAGE 2: SEGMENT EXPLORER
# -------------------------------------------------------------------
elif page == "🔍 Customer Segment Explorer":
    st.markdown('<div class="main-title">Interactive Customer Segment Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Filter historical campaign data dynamically to inspect segment conversion yield</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_jobs = st.multiselect("Select Job Types:", options=df['job'].unique(), default=df['job'].unique())
    with col2:
        sel_edu = st.multiselect("Select Education Level:", options=df['education'].unique(), default=df['education'].unique())
    with col3:
        sel_housing = st.selectbox("Housing Loan Filter:", options=['All', 'Yes', 'No'], index=0)
        
    filtered = df[(df['job'].isin(sel_jobs)) & (df['education'].isin(sel_edu))]
    if sel_housing != 'All':
        filtered = filtered[filtered['housing'] == sel_housing.lower()]
        
    st.markdown("---")
    f_total = len(filtered)
    f_sub = int(filtered['target'].sum())
    f_rate = (f_sub / f_total * 100) if f_total > 0 else 0.0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Segment Size", f"{f_total:,} customers")
    m2.metric("Segment Subscriptions", f"{f_sub:,}")
    m3.metric("Segment Conversion Rate", f"{f_rate:.2f}%", delta=f"{f_rate - 11.52:.2f}% vs Baseline")
    
    st.dataframe(filtered[['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'campaign', 'poutcome', 'y']].head(20), width='stretch')

# -------------------------------------------------------------------
# PAGE 3: PRE-CONTACT PREDICTOR
# -------------------------------------------------------------------
elif page == "🔮 Pre-Contact Propensity Predictor":
    st.markdown('<div class="main-title">Pre-Contact Term Deposit Response Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Score and rank customer prospects before dialing their phone numbers</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="leak-banner">
        ⚠️ <strong>Methodological Governance (Leakage Avoidance):</strong><br>
        In accordance with our experimental findings, <strong>Call Duration is strictly excluded</strong> from this prospective model.
        All inputs represent information known to the bank <em>before</em> initiating contact.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_age = st.number_input("Customer Age:", min_value=18, max_value=100, value=35)
            in_job = st.selectbox("Occupation:", options=df['job'].unique(), index=0)
            in_marital = st.selectbox("Marital Status:", options=df['marital'].unique(), index=1)
            in_edu = st.selectbox("Education Level:", options=df['education'].unique(), index=1)
            
        with col2:
            in_balance = st.number_input("Average Yearly Balance (€):", min_value=-5000, max_value=100000, value=1500)
            in_default = st.selectbox("Credit Default History:", options=['no', 'yes'])
            in_housing = st.selectbox("Has Housing Loan / Mortgage:", options=['no', 'yes'])
            in_loan = st.selectbox("Has Personal Debt Loan:", options=['no', 'yes'])
            
        with col3:
            in_contact = st.selectbox("Outreach Communication Channel:", options=df['contact'].unique(), index=0)
            in_month = st.selectbox("Scheduled Contact Month:", options=df['month'].unique(), index=4)
            in_day = st.slider("Scheduled Day of Month:", 1, 31, 15)
            in_campaign = st.number_input("Planned Contact Attempts (Campaign):", min_value=1, max_value=20, value=1)
            in_pdays = st.number_input("Days Since Prior Campaign (-1 if never):", min_value=-1, max_value=1000, value=-1)
            in_previous = st.number_input("Number of Contacts Before This Campaign:", min_value=0, max_value=50, value=0)
            in_poutcome = st.selectbox("Outcome of Previous Campaign:", options=['unknown', 'failure', 'other', 'success'], index=0)
            
        submit_btn = st.form_submit_button("Compute Pre-Contact Subscription Propensity")
        
    if submit_btn:
        # Derive engineered features
        age_grp = '<30' if in_age < 30 else ('30-39' if in_age < 40 else ('40-49' if in_age < 50 else ('50-59' if in_age < 60 else '60+')))
        if in_balance < 0: bal_grp = 'Negative'
        elif in_balance <= 500: bal_grp = 'Low (0-500)'
        elif in_balance <= 2000: bal_grp = 'Medium (501-2000)'
        else: bal_grp = 'High (>2000)'
        prev_con = 'No' if in_pdays == -1 else 'Yes'
        if in_campaign == 1: freq_grp = '1 contact'
        elif in_campaign <= 3: freq_grp = '2-3 contacts'
        else: freq_grp = '4+ contacts'
        
        row_dict = {
            'age': in_age, 'balance': in_balance, 'day': in_day, 'campaign': in_campaign,
            'pdays': in_pdays, 'previous': in_previous,
            'job': in_job, 'marital': in_marital, 'education': in_edu, 'default': in_default,
            'housing': in_housing, 'loan': in_loan, 'contact': in_contact, 'month': in_month,
            'poutcome': in_poutcome, 'age_group': age_grp, 'balance_group': bal_grp,
            'previously_contacted': prev_con, 'contact_frequency': freq_grp
        }
        input_df = pd.DataFrame([row_dict])
        
        # Predict probability
        prob = model_pipe.predict_proba(input_df)[0, 1]
        pct = prob * 100
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Estimated Subscription Propensity", f"{pct:.1f}%")
            if pct >= 35.0:
                st.success("🟢 **High Priority Lead** — Recommended for immediate outbound outreach by senior advisor.")
            elif pct >= 15.0:
                st.warning("🟡 **Medium Priority Lead** — Standard campaign routing.")
            else:
                st.error("🔴 **Low Priority Lead** — Recommend digital nurture before placing direct calls.")
                
        with res_col2:
            st.write("**Key Profile Drivers Detected:**")
            st.write(f"- **Prior Campaign Status:** `{in_poutcome}` (Historical conversions provide strong predictive lift).")
            st.write(f"- **Credit Commitment:** Housing Loan: `{in_housing}`, Personal Debt: `{in_loan}`.")
            st.write(f"- **Capital Reserve Tier:** `{bal_grp}` (€{in_balance:,.0f}).")
            st.write(f"- **Contact Plan:** `{freq_grp}` planned contacts.")

# -------------------------------------------------------------------
# PAGE 4: PROJECT INFO
# -------------------------------------------------------------------
elif page == "ℹ️ Academic Project Info":
    st.markdown('<div class="main-title">Academic Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    **Project Title:** Bank Campaign Analytics and Term Deposit Response Prediction Using Machine Learning  
    **Student Name:** Mohan Kumar M  
    **Internship Program:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026  
    **Host Organization:** BharatCares in association with All India Council for Technical Education (AICTE)  
    
    ---
    
    ### Academic Objectives
    1. Apply Python data analytics, statistical visualization, and machine learning to a real-world commercial banking dataset.
    2. Prevent data leakage by isolating the post-contact variable `duration`.
    3. Compare Logistic Regression and Random Forest classifiers across Accuracy, Precision, Recall, F1-score, and ROC-AUC.
    4. Provide evidence-based business recommendations for telemarketing campaign managers.
    
    ### Official Dataset Citation
    - **Source:** UCI Machine Learning Repository — Bank Marketing Dataset (`bank.csv`)
    - **Citation:** Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22-31.
    """)
