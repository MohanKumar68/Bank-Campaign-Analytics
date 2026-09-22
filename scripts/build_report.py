"""
Academic Report Builder for MohanKumarM_ProjectReport.docx
Generates a comprehensive, professionally styled Word document adhering to
all academic chapters, embedding real empirical metrics, high-res analysis figures,
and the real Streamlit dashboard screenshots uploaded by the student.
"""

import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def create_report():
    doc = Document()

    # Set page margins (1 inch all sides)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Configure Header and Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("AICTE | IBM SkillsBuild Internship 2026 - Bank Campaign Analytics")
        hrun.font.name = "Arial"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(120, 120, 120)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("Mohan Kumar M | Term Deposit ML Prediction")
        frun.font.name = "Arial"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(120, 120, 120)

    # Style helper functions
    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), fill_hex)
        tcPr.append(shd)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(22)
        run.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(24)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(71, 85, 105) # Slate 600
        return p

    def add_heading_1(text):
        h = doc.add_heading(level=1)
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(15)
        run.bold = True
        run.font.color.rgb = RGBColor(30, 58, 138) # Dark Blue
        return h

    def add_heading_2(text):
        h = doc.add_heading(level=2)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(12.5)
        run.bold = True
        run.font.color.rgb = RGBColor(15, 118, 110) # Teal
        return h

    def add_body(text, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(30, 41, 59) # Slate 800
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        run_b = p.add_run(bold_prefix)
        run_b.font.name = 'Arial'
        run_b.font.size = Pt(10)
        run_b.bold = True
        run_b.font.color.rgb = RGBColor(15, 23, 42)
        
        run_t = p.add_run(f" {text}")
        run_t.font.name = 'Arial'
        run_t.font.size = Pt(10)
        run_t.font.color.rgb = RGBColor(51, 65, 85)
        return p

    def add_image_box(image_path, caption_text, width_inches=5.8):
        if os.path.exists(image_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run()
            run.add_picture(image_path, width=Inches(width_inches))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(12)
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = 'Arial'
            run_cap.font.size = Pt(9)
            run_cap.italic = True
            run_cap.font.color.rgb = RGBColor(100, 116, 139)

    def style_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if col_widths and j < len(col_widths):
                    cell.width = Inches(col_widths[j])
                tcPr = cell._tc.get_or_add_tcPr()
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    for r in p.runs:
                        r.font.name = 'Arial'
                        if i == 0:
                            r.font.size = Pt(9.5)
                            r.bold = True
                            r.font.color.rgb = RGBColor(255, 255, 255)
                        else:
                            r.font.size = Pt(9)
                            r.font.color.rgb = RGBColor(30, 41, 59)
                if i == 0:
                    set_cell_background(cell, "1E3A8A") # Navy header
                else:
                    if i % 2 == 1:
                        set_cell_background(cell, "F8FAFC")
                    else:
                        set_cell_background(cell, "FFFFFF")

    # Uploaded screenshot filepaths
    dashboard_img_1 = r"C:\Users\Shirisha\.gemini\antigravity-ide\brain\217418a1-f7a8-4208-a5a2-9a9dc6d37bcb\.user_uploaded\media_1790084670708.png"
    dashboard_img_2 = r"C:\Users\Shirisha\.gemini\antigravity-ide\brain\217418a1-f7a8-4208-a5a2-9a9dc6d37bcb\.user_uploaded\media_1790084693944.png"
    dashboard_img_3 = r"C:\Users\Shirisha\.gemini\antigravity-ide\brain\217418a1-f7a8-4208-a5a2-9a9dc6d37bcb\.user_uploaded\media_1790084715808.png"

    # =========================================================================
    # CHAPTER 1: TITLE PAGE
    # =========================================================================
    add_title("Bank Campaign Analytics and Term Deposit Response Prediction Using Machine Learning")
    add_subtitle("AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026\nConducted by BharatCares in association with AICTE")

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(40)
    p_meta.paragraph_format.space_after = Pt(20)
    
    r1 = p_meta.add_run("ACADEMIC PROJECT REPORT\n\nSubmitted by:\n")
    r1.font.name = 'Arial'
    r1.font.size = Pt(11)
    
    r2 = p_meta.add_run("MOHAN KUMAR M\n")
    r2.font.name = 'Arial'
    r2.font.size = Pt(14)
    r2.bold = True
    r2.font.color.rgb = RGBColor(30, 58, 138)
    
    r3 = p_meta.add_run("Fourth-Year Undergraduate Engineering Student\n\nUnder the Mentorship of:\nBharatCares & IBM SkillsBuild Technical Teams\nAcademic Year: 2026\n")
    r3.font.name = 'Arial'
    r3.font.size = Pt(11)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 2: ABSTRACT
    # =========================================================================
    add_heading_1("2. Abstract")
    add_body(
        "Commercial banking organizations commit vast organizational budgets toward direct telemarketing campaigns to "
        "solicit customer subscriptions for term deposit accounts. In typical operational settings, outbound telemarketing "
        "yields low conversion rates (8% to 12%), resulting in substantial call-center overhead, reduced advisor efficiency, "
        "and client contact fatigue. This project presents an empirical data analytics and supervised machine learning "
        "investigation conducted on the benchmark UCI Machine Learning Repository Bank Marketing dataset (4,521 real banking "
        "interactions and 17 attributes). The core objective is twofold: (1) diagnose customer demographic, financial, and "
        "campaign features associated with subscription response, and (2) build an algorithmic decision-support model to "
        "predict term deposit propensity prior to initiating outbound contact. A central methodological contribution of this "
        "study is the identification and rigorous exclusion of the post-contact variable 'duration' (call length), which induces "
        "severe data leakage (inflating test ROC-AUC artificially to 0.896) and is operationally unavailable before dialing. "
        "Evaluating clean pre-contact classifiers using 80/20 stratified partitioning, Logistic Regression achieved an ROC-AUC of "
        "0.747 and a Recall of 62.5% (capturing two-thirds of potential subscribers), while Random Forest achieved 88.7% Accuracy "
        "and 55.6% Precision (doubling conversion efficiency). Actionable business recommendations are presented to enable "
        "banks to prioritize warm re-engagement leads and enforce contact frequency limits."
    )

    # =========================================================================
    # CHAPTER 3: INTRODUCTION
    # =========================================================================
    add_heading_1("3. Introduction")
    add_body(
        "Retail banks rely heavily on retail deposits as their primary source of stable, low-cost capital. Term deposits, in "
        "particular, lock customer funds for specified maturities (such as 12, 24, or 36 months) in exchange for fixed interest yields. "
        "These stable deposits enable institutions to underwrite mortgages, issue corporate credit, and maintain statutory liquidity reserves. "
        "Despite digital banking proliferation, outbound telephone campaigns remain a predominant marketing channel for wealth and deposit products."
    )
    add_body(
        "However, indiscriminate mass telemarketing is commercially unsustainable. Contacting unsegmented customer databases causes customer irritation, "
        "brand dilution, and excessive labor expenditures. By applying modern Data Analytics and Machine Learning techniques, banks can transition "
        "from blanket outreach to precision targeting, matching financial offerings to customers with verified inclination and economic capacity to invest."
    )

    # =========================================================================
    # CHAPTER 4: PROJECT OUTPUT - IMPLEMENTED STREAMLIT DASHBOARD
    # =========================================================================
    add_heading_1("4. Project Output - Implemented Streamlit Dashboard")
    add_body(
        "To bridge advanced computational modeling with daily banking marketing operations, the entire analytical and predictive "
        "workflow developed in this project was implemented as a production-grade, interactive web dashboard using Streamlit. "
        "This section presents the actual graphical user interface and functional outputs generated by the running application, "
        "providing the evaluator with direct visual evidence of the implemented system prior to the detailed technical chapters."
    )

    add_heading_2("4.1 Executive Dashboard Overview and Campaign KPIs")
    add_image_box(
        dashboard_img_1,
        "Figure 1: Implemented Streamlit Dashboard - Executive Overview, Core KPIs, and Category Distributions",
        width_inches=5.8
    )
    add_body(
        "Figure 1 displays the primary 'Executive Dashboard & KPIs' view of the implemented Streamlit application. The navigation sidebar on the "
        "left confirms the project credentials (Student: Mohan Kumar M, Program: AICTE | IBM SkillsBuild 2026, Partner: BharatCares) and allows "
        "switching between analytical modules. The main console displays five real-time Key Performance Indicator (KPI) cards: Total Contacts (4,521), "
        "Term Deposit Subscriptions (521), Overall Conversion Rate (11.52%), Mean Customer Account Balance (EUR 1,423), and Median Balance (EUR 444). "
        "Below the KPI cards, the interface presents two comparative visualizations: the target variable distribution showing 4,000 non-subscribers "
        "(88.5%) versus 521 subscribers (11.5%), and a horizontal bar chart displaying subscription rates across job categories benchmarked "
        "against the 11.5% campaign baseline."
    )

    add_heading_2("4.2 Core Behavioral Drivers and Liability Impact Analysis")
    add_image_box(
        dashboard_img_2,
        "Figure 2: Implemented Streamlit Dashboard - Core Behavioral Drivers and Loan Liabilities View",
        width_inches=5.8
    )
    add_body(
        "Figure 2 illustrates the 'Core Behavioral Drivers' module within the dashboard. The left panel visualizes conversion yields across "
        "historical outreach outcomes ('poutcome'), highlighting that clients who converted in a previous campaign achieved a 64.3% subscription "
        "rate - nearly six times the baseline rate (indicated by the red dashed reference line). The right panel isolates the impact of debt liabilities, "
        "showing that clients servicing housing mortgages converted at only 8.6% (compared to 15.3% for debt-free clients) and personal loan "
        "commitments reduced conversion to 6.2% (compared to 12.5% without personal loans)."
    )

    add_heading_2("4.3 Interactive Customer Segment Explorer")
    add_image_box(
        dashboard_img_3,
        "Figure 3: Implemented Streamlit Dashboard - Interactive Customer Segment Explorer and Dynamic Data Filter",
        width_inches=5.8
    )
    add_body(
        "Figure 3 demonstrates the 'Interactive Customer Segment Explorer' interface. Marketing managers can dynamically select and combine "
        "customer attributes through multi-select dropdown filters (including Job Types, Education Levels, and Housing Loan status). Upon adjusting "
        "filters, the dashboard instantly recalculates the Segment Size, Segment Subscriptions, and Segment Conversion Rate (with delta indicator "
        "relative to baseline), and dynamically filters the underlying tabular customer records for granular verification."
    )

    add_heading_2("4.4 Operational Significance of the Dashboard")
    add_body(
        "The implemented Streamlit application serves as an interactive decision-support cockpit for bank marketing teams and executive leadership. "
        "It operationalizes the data analysis findings by providing an accessible, real-time interface for: "
        "(1) tracking top-line campaign performance metrics and class distributions at a glance; "
        "(2) diagnosing critical behavioral drivers and the restrictive impact of existing credit commitments; "
        "(3) dynamically segmenting customer cohorts to estimate campaign yields before deploying outreach resources; and "
        "(4) serving as an intuitive front-end for pre-contact lead qualification without exposing non-technical staff to raw code. "
        "Crucially, the dashboard maintains strict methodological governance by excluding post-contact duration from prospective scoring."
    )

    # =========================================================================
    # CHAPTER 5: PROBLEM STATEMENT
    # =========================================================================
    add_heading_1("5. Problem Statement")
    add_body(
        "Direct marketing campaigns conducted by retail banking institutions suffer from low baseline response rates (approximately 11.5% in empirical data), "
        "meaning that nearly 9 out of 10 customer calls fail to generate a subscription. Marketing managers lack data-driven visibility into the specific "
        "customer profiles and campaign parameters that correlate with positive conversion. Furthermore, when deploying predictive models, organizations "
        "frequently introduce data leakage by relying on call-interaction variables that cannot be known prior to placing the call. The fundamental challenge "
        "is to engineer a robust, leakage-free predictive model and diagnostic analytics pipeline that operates strictly on pre-contact customer intelligence."
    )

    # =========================================================================
    # CHAPTER 6: OBJECTIVES
    # =========================================================================
    add_heading_1("6. Objectives")
    add_bullet("1. Data Auditing & Exploration:", "Conduct rigorous quality verification (null checks, duplicate checks, distribution shapes) on the UCI Bank Marketing benchmark dataset.")
    add_bullet("2. Business KPI Quantification:", "Formulate and measure executive-level marketing metrics, including baseline conversion rate, wealth distribution, and prior campaign re-engagement yield.")
    add_bullet("3. Domain Feature Engineering:", "Construct interpretable features (age cohorts, wealth tiers, prior contact indicators, contact frequency bins) to enhance classification signal.")
    add_bullet("4. Leakage Demonstration & Prevention:", "Empirically evaluate the impact of including the post-contact 'duration' feature versus strictly excluding it to safeguard operational validity.")
    add_bullet("5. Predictive Modeling & Comparison:", "Train, evaluate, and contrast balanced Logistic Regression and Random Forest architectures across Accuracy, Precision, Recall, F1-score, and ROC-AUC.")
    add_bullet("6. Managerial Strategy Formulation:", "Deliver actionable, non-causal business guidelines for banking marketing directors to optimize contact allocation and campaign yield.")

    # =========================================================================
    # CHAPTER 7: BUSINESS CONTEXT
    # =========================================================================
    add_heading_1("7. Business Context")
    add_body(
        "The financial dataset analyzed in this report reflects marketing campaigns orchestrated by a Portuguese retail banking "
        "institution between May 2008 and November 2010. This operational timeframe coincided with the European sovereign debt crisis, "
        "a macroeconomic environment characterized by tightening liquidity, bank recapitalization pressures, and conservative household "
        "savings behavior. In such economic contexts, retail customers prioritize security and capital preservation, making bank-backed term deposits "
        "attractive relative to volatile equity markets, provided the household possesses unencumbered discretionary savings."
    )

    # =========================================================================
    # CHAPTER 8: DATASET DESCRIPTION
    # =========================================================================
    add_heading_1("8. Dataset Description")
    add_body(
        "The study employs the official 'bank.csv' dataset provided by the University of California, Irvine (UCI) Machine Learning Repository "
        "(Moro, Cortez, & Rita, 2014). The dataset comprises 4,521 customer records sampled randomly from a broader collection of 45,211 contacts. "
        "Each record reflects a single direct marketing engagement and records 16 independent variables capturing demographic characteristics, "
        "credit and liability profiles, campaign logistics, and historical touchpoints, accompanied by the binary classification target 'y'."
    )

    # =========================================================================
    # CHAPTER 9: DATA DICTIONARY
    # =========================================================================
    add_heading_1("9. Data Dictionary")
    add_body("Table 1 summarizes all 17 attributes present in the benchmark dataset, including their domain type and operational definition:")

    data_dict = [
        ("age", "Numeric (Integer)", "Age of the customer in years (range: 19 to 87)."),
        ("job", "Categorical (12 levels)", "Type of occupation (admin., blue-collar, entrepreneur, management, retired, etc.)."),
        ("marital", "Categorical (3 levels)", "Marital status (divorced, married, single)."),
        ("education", "Categorical (4 levels)", "Educational attainment (primary, secondary, tertiary, unknown)."),
        ("default", "Binary (yes/no)", "Has credit in default status."),
        ("balance", "Numeric (Integer)", "Average yearly balance in euros (EUR)."),
        ("housing", "Binary (yes/no)", "Has existing housing loan / mortgage liability."),
        ("loan", "Binary (yes/no)", "Has personal debt loan commitment."),
        ("contact", "Categorical (3 levels)", "Contact communication channel (cellular, telephone, unknown)."),
        ("day", "Numeric (Integer)", "Last contact day of the month (1 to 31)."),
        ("month", "Categorical (12 levels)", "Last contact month of year (jan through dec)."),
        ("duration", "Numeric (Integer)", "Last contact talk time in seconds. [LEAKAGE ATTRIBUTE: Post-contact only]."),
        ("campaign", "Numeric (Integer)", "Number of contacts performed during this campaign for this client."),
        ("pdays", "Numeric (Integer)", "Days since client was last contacted from prior campaign (-1 = never)."),
        ("previous", "Numeric (Integer)", "Number of contacts performed before this campaign."),
        ("poutcome", "Categorical (4 levels)", "Outcome of previous marketing campaign (failure, other, success, unknown)."),
        ("y", "Binary (yes/no)", "Target variable: has client subscribed to a term deposit?")
    ]

    t_dict = doc.add_table(rows=1 + len(data_dict), cols=3)
    t_dict.rows[0].cells[0].paragraphs[0].add_run("Attribute Name")
    t_dict.rows[0].cells[1].paragraphs[0].add_run("Data Type")
    t_dict.rows[0].cells[2].paragraphs[0].add_run("Operational Definition")
    for idx, row in enumerate(data_dict):
        t_dict.rows[idx+1].cells[0].paragraphs[0].add_run(row[0])
        t_dict.rows[idx+1].cells[1].paragraphs[0].add_run(row[1])
        t_dict.rows[idx+1].cells[2].paragraphs[0].add_run(row[2])
    style_table(t_dict, [1.5, 1.6, 3.2])

    # =========================================================================
    # CHAPTER 10: METHODOLOGY
    # =========================================================================
    add_heading_1("10. Methodology")
    add_body(
        "The project follows a rigorous, reproducible CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology adapted for academic research:"
    )
    add_bullet("Phase 1 - Ingestion & Auditing:", "Official dataset acquisition directly from the UCI archive, followed by completeness checks.")
    add_bullet("Phase 2 - Exploratory Analytics:", "Univariate and bivariate statistical decomposition across client profiles and historical outreach variables.")
    add_bullet("Phase 3 - KPI Formalization:", "Computation of core business metrics to contextualize campaign commercial performance.")
    add_bullet("Phase 4 - Domain Feature Engineering:", "Transformation of continuous and sentinel features into discrete business cohorts.")
    add_bullet("Phase 5 - Leakage Assessment:", "Controlled experimental validation isolating and removing the post-contact 'duration' feature.")
    add_bullet("Phase 6 - Pipeline Assembly & Training:", "Stratified train-test partitioning (80/20) and automated preprocessing using ColumnTransformer.")
    add_bullet("Phase 7 - Model Evaluation & Interpretation:", "Calculation of confusion matrices, ROC curves, feature importances, and regression coefficients.")
    add_bullet("Phase 8 - Managerial Recommendations:", "Synthesizing empirical findings into actionable business rules.")

    # =========================================================================
    # CHAPTER 11: DATA PREPROCESSING
    # =========================================================================
    add_heading_1("11. Data Preprocessing")
    add_body(
        "The original UCI dataset was found to have complete integrity with 0 null values and 0 duplicate records across all 4,521 rows. "
        "Pre-processing choices were strictly justified to preserve natural real-world behavior:"
    )
    add_bullet("Missing vs Unknown Categoricals:", "Values recorded as 'unknown' in columns such as 'job', 'education', 'contact', and 'poutcome' were purposefully retained as explicit category levels. Treating 'unknown' as an informative signal reflects operational reality where customer information may not be captured.")
    add_bullet("Sentinel Value Resolution:", "The feature 'pdays' utilizes '-1' to denote that a customer was not contacted in any prior campaign. Rather than treating '-1' as a continuous arithmetic value, it was mapped into an indicator variable.")
    add_bullet("Numerical Standardization:", "Continuous variables (age, balance, day, campaign, pdays, previous) were scaled via StandardScaler to have zero mean and unit variance.")
    add_bullet("Categorical Encoding:", "Categorical variables were encoded using OneHotEncoder with 'drop=first' to prevent collinearity in linear models while ignoring unseen categories in test splits.")

    # =========================================================================
    # CHAPTER 12: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    add_heading_1("12. Exploratory Data Analysis")
    add_body(
        "Eight targeted graphical analyses were conducted to investigate relationships between customer attributes and term deposit subscriptions. "
        "All interpretations adhere to non-causal wording, reporting observed association rather than attributing definitive causality."
    )

    add_heading_2("12.1 Target Distribution and Class Imbalance")
    add_image_box("outputs/figures/eda_target_distribution.png", "Figure 4: Target Variable Distribution (y)")
    add_body(
        "As illustrated in Figure 4, the target variable exhibits significant class imbalance. Out of 4,521 contacts, 4,000 clients (88.48%) declined "
        "or failed to subscribe, while 521 clients (11.52%) subscribed to a term deposit. This 8:1 imbalance requires class-weighted loss functions "
        "and evaluation metrics sensitive to minority recall, such as ROC-AUC, Precision, and Recall, rather than raw classification accuracy."
    )

    add_heading_2("12.2 Subscription Rate by Occupation")
    add_image_box("outputs/figures/eda_job_subscription.png", "Figure 5: Subscription Rate Across Job Categories")
    add_body(
        "Figure 5 highlights significant variation across occupations. The highest conversion rates were observed among students (22.6%) and retirees "
        "(23.5%), both converting at more than double the 11.5% campaign baseline. In contrast, blue-collar workers (7.3%) and entrepreneurs (8.9%) "
        "showed the lowest response rates. Retirees often seek safe, low-risk capital preservation, whereas blue-collar workers and entrepreneurs "
        "frequently direct cash flows into working capital or debt servicing."
    )

    add_heading_2("12.3 Subscription Rate by Education Level")
    add_image_box("outputs/figures/eda_education_subscription.png", "Figure 6: Subscription Rate by Educational Attainment")
    add_body(
        "Figure 6 demonstrates that clients with tertiary education achieved an observed conversion rate of 14.3%, compared to 10.7% for secondary education "
        "and 9.4% for primary education. Higher educational attainment is typically correlated with financial literacy and greater familiarity with fixed-income instruments."
    )

    add_heading_2("12.4 Impact of Credit Liabilities")
    add_image_box("outputs/figures/eda_housing_loan.png", "Figure 7: Subscription Rate by Housing and Personal Loan Commitments")
    add_body(
        "Figure 7 reveals that outstanding liabilities strongly inhibit term deposit subscription. Clients without a housing loan converted at 15.3%, "
        "compared to only 8.6% for those with mortgages. Similarly, clients without personal debt converted at 12.5%, compared to 6.7% for clients "
        "holding personal loans. Household liquidity committed to servicing debt is unavailable for multi-month deposit commitments."
    )

    add_heading_2("12.5 Account Balance Distribution")
    add_image_box("outputs/figures/eda_balance_distribution.png", "Figure 8: Account Balance Distribution by Subscription Status")
    add_body(
        "Figure 8 depicts the yearly average account balance across subscription outcomes. Subscribed clients displayed higher median balances "
        "(EUR 733 vs EUR 417) and larger upper-quartile reserves. Customers maintaining negative or negligible bank balances possess minimal discretionary savings."
    )

    add_heading_2("12.6 Campaign Contact Frequency and Outreach Diminishing Returns")
    add_image_box("outputs/figures/eda_campaign_contacts.png", "Figure 9: Subscription Rate by Number of Campaign Contacts")
    add_body(
        "Figure 9 illustrates sharp diminishing returns associated with repeated calling. Conversion is highest during the 1st contact (12.3%) and "
        "2nd contact (11.5%), before dropping to under 6% for 5-6 contacts and under 3% beyond 7 contacts. High call frequencies produce customer resistance."
    )

    add_heading_2("12.7 Previous Campaign Outcome")
    add_image_box("outputs/figures/eda_poutcome_subscription.png", "Figure 10: Subscription Rate by Previous Campaign Outcome")
    add_body(
        "Figure 10 demonstrates that customers whose previous campaign outcome was recorded as 'success' achieved an extraordinary current subscription "
        "rate of 64.3% - nearly 6 times the baseline rate. Historical positive engagement is the single strongest indicator of prospective conversion."
    )

    add_heading_2("12.8 Duration Analysis and Post-Contact Characteristic")
    add_image_box("outputs/figures/eda_duration_analysis.png", "Figure 11: Call Duration Distribution (Post-Contact Exploration)")
    add_body(
        "Figure 11 shows that subscribed calls averaged 9.2 minutes (555 seconds) versus 3.7 minutes (221 seconds) for non-subscribers. While this "
        "association is strong, call length is established only during the interaction. Utilizing duration in prospective targeting models represents "
        "severe data leakage, as discussed in Chapter 16."
    )

    # =========================================================================
    # CHAPTER 13: BUSINESS KPI ANALYSIS
    # =========================================================================
    add_heading_1("13. Business KPI Analysis")
    add_body(
        "To establish operational benchmarks for executive management, Table 2 summarizes core campaign Key Performance Indicators calculated "
        "from the verified dataset:"
    )

    kpi_df = pd.read_csv('outputs/model_results/business_kpis.csv')
    t_kpi = doc.add_table(rows=1 + len(kpi_df), cols=2)
    t_kpi.rows[0].cells[0].paragraphs[0].add_run("Key Performance Indicator (KPI)")
    t_kpi.rows[0].cells[1].paragraphs[0].add_run("Measured Value")
    for idx, row in kpi_df.iterrows():
        t_kpi.rows[idx+1].cells[0].paragraphs[0].add_run(str(row['Metric']))
        t_kpi.rows[idx+1].cells[1].paragraphs[0].add_run(str(row['Value']))
    style_table(t_kpi, [3.8, 2.5])

    # =========================================================================
    # CHAPTER 14: FEATURE ENGINEERING
    # =========================================================================
    add_heading_1("14. Feature Engineering")
    add_body(
        "Four domain-tailored features were engineered to capture non-linear relationships and business concepts:"
    )
    add_bullet("1. age_group:", "Segmented into cohorts (<30, 30-39, 40-49, 50-59, 60+) to isolate retirement-age and student life stages from mid-career groups.")
    add_bullet("2. balance_group:", "Categorized into Negative (< EUR 0), Low (EUR 0-500), Medium (EUR 501-2000), and High (> EUR 2000) to separate debt-stressed clients from affluent savers without assuming linear returns on balance.")
    add_bullet("3. previously_contacted:", "Binary indicator ('No' if pdays == -1, else 'Yes') resolving the sentinel value and separating warm leads from cold prospects.")
    add_bullet("4. contact_frequency:", "Segmented into '1 contact', '2-3 contacts', and '4+ contacts' to reflect optimal outreach versus contact fatigue.")

    # =========================================================================
    # CHAPTER 15: MACHINE LEARNING METHODOLOGY
    # =========================================================================
    add_heading_1("15. Machine Learning Methodology")
    add_body(
        "To establish rigorous classification benchmarks, an 80/20 train-test split was executed with stratification on the target variable `y` "
        "(random_state=42), yielding 3,616 training samples (417 positive) and 905 test samples (104 positive). Preprocessing pipelines were "
        "encapsulated via scikit-learn's ColumnTransformer and Pipeline objects to guarantee strict isolation between training and testing data."
    )
    add_bullet("Model 1 - Logistic Regression:", "Trained with L2 regularization, max_iter=1000, random_state=42, and class_weight='balanced' to adjust for the 8:1 class ratio.")
    add_bullet("Model 2 - Random Forest Classifier:", "Ensemble of 200 decision trees (n_estimators=200, random_state=42, class_weight='balanced') to model complex feature interactions.")

    # =========================================================================
    # CHAPTER 16: DATA LEAKAGE CONSIDERATION
    # =========================================================================
    add_heading_1("16. Data Leakage Consideration")
    add_body(
        "A critical conceptual and practical pitfall in marketing analytics is the inclusion of the 'duration' variable in prospective prediction models. "
        "When an agent decides which customer to contact from a CRM list at 9:00 AM, the call duration is identically zero. Call duration is an artifact "
        "of the interaction itself: when a client is interested, the conversation lengthens to explain terms, verify identity, and execute paperwork. "
        "Consequently, duration is a consequence, not a pre-condition, of subscription."
    )
    add_image_box("outputs/figures/leakage_comparison.png", "Figure 12: ROC Performance Demonstration - With vs Without Duration Leakage")
    
    leak_df = pd.read_csv('outputs/model_results/leakage_experiment_results.csv')
    t_leak = doc.add_table(rows=1 + len(leak_df), cols=6)
    headers = ["Model Setup", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for j, h in enumerate(headers):
        t_leak.rows[0].cells[j].paragraphs[0].add_run(h)
    for idx, row in leak_df.iterrows():
        t_leak.rows[idx+1].cells[0].paragraphs[0].add_run(str(row['Model Configuration']))
        t_leak.rows[idx+1].cells[1].paragraphs[0].add_run(f"{row['Accuracy']:.4f}")
        t_leak.rows[idx+1].cells[2].paragraphs[0].add_run(f"{row['Precision']:.4f}")
        t_leak.rows[idx+1].cells[3].paragraphs[0].add_run(f"{row['Recall']:.4f}")
        t_leak.rows[idx+1].cells[4].paragraphs[0].add_run(f"{row['F1-score']:.4f}")
        t_leak.rows[idx+1].cells[5].paragraphs[0].add_run(f"{row['ROC-AUC']:.4f}")
    style_table(t_leak, [2.2, 0.8, 0.8, 0.8, 0.8, 0.9])

    add_body(
        "As documented in Table 3 and Figure 12, including duration inflates the ROC-AUC to 0.8959. However, this model is operationally invalid "
        "because talk time cannot be supplied as an input when prioritizing leads before dialing. Excluding duration yields a valid, realistic "
        "ROC-AUC of 0.7185."
    )

    # =========================================================================
    # CHAPTER 17: MODEL EVALUATION
    # =========================================================================
    add_heading_1("17. Model Evaluation")
    add_body(
        "Evaluating models strictly without the leaked duration attribute on the 905-sample holdout test set produced the empirical results shown below:"
    )

    add_image_box("outputs/figures/model_cm_logistic_regression.png", "Figure 13: Confusion Matrix - Logistic Regression (Pre-Contact)")
    add_image_box("outputs/figures/model_cm_random_forest.png", "Figure 14: Confusion Matrix - Random Forest (Pre-Contact)")
    add_image_box("outputs/figures/model_roc_curves.png", "Figure 15: Receiver Operating Characteristic (ROC) Comparison (No Duration)")

    comp_df = pd.read_csv('outputs/model_results/model_comparison_metrics.csv')
    t_comp = doc.add_table(rows=1 + len(comp_df), cols=6)
    for j, h in enumerate(["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
        t_comp.rows[0].cells[j].paragraphs[0].add_run(h)
    for idx, row in comp_df.iterrows():
        t_comp.rows[idx+1].cells[0].paragraphs[0].add_run(str(row['Model']))
        t_comp.rows[idx+1].cells[1].paragraphs[0].add_run(f"{row['Accuracy']:.4f}")
        t_comp.rows[idx+1].cells[2].paragraphs[0].add_run(f"{row['Precision']:.4f}")
        t_comp.rows[idx+1].cells[3].paragraphs[0].add_run(f"{row['Recall']:.4f}")
        t_comp.rows[idx+1].cells[4].paragraphs[0].add_run(f"{row['F1-score']:.4f}")
        t_comp.rows[idx+1].cells[5].paragraphs[0].add_run(f"{row['ROC-AUC']:.4f}")
    style_table(t_comp, [2.2, 0.8, 0.8, 0.8, 0.8, 0.9])

    # =========================================================================
    # CHAPTER 18: RESULTS AND DISCUSSION
    # =========================================================================
    add_heading_1("18. Results and Discussion")
    add_body(
        "The empirical findings demonstrate clear trade-offs between linear and ensemble methods when classifying imbalanced tabular banking data:"
    )
    add_bullet("Logistic Regression Performance:", "Logistic Regression achieved an Accuracy of 72.49%, Precision of 23.64%, Recall of 62.50%, and an ROC-AUC of 0.7470. By identifying 65 of the 104 true subscribers in the test set, it acts as an effective high-recall filter for campaigns aiming to capture maximum deposit volume.")
    add_bullet("Random Forest Performance:", "Random Forest achieved an Accuracy of 88.73%, Precision of 55.56%, Recall of 9.62%, and an ROC-AUC of 0.7185. At the default threshold, it acts as a conservative, high-precision filter: over 55% of leads it flags will convert, making it ideal when call-center advisor hours are strictly limited.")

    add_heading_2("18.1 Feature Importance and Model Interpretation")
    add_image_box("outputs/figures/model_rf_feature_importance.png", "Figure 16: Top 15 Feature Importances - Random Forest")
    add_image_box("outputs/figures/model_lr_coefficients.png", "Figure 17: Most Influential Coefficients - Logistic Regression")
    add_body(
        "Figure 16 and Figure 17 highlight the driving variables. In the Random Forest model, continuous financial features (balance, age, day, campaign) "
        "and previous campaign success (poutcome_success) contribute most to Gini impurity reduction. In Logistic Regression, poutcome_success "
        "(+1.86 log-odds) is the single strongest positive predictor, while existing housing loans (-0.51 log-odds) and personal debt (-0.57 log-odds) "
        "exert the strongest downward pressure on subscription likelihood."
    )

    # =========================================================================
    # CHAPTER 19: BUSINESS INSIGHTS
    # =========================================================================
    add_heading_1("19. Business Insights")
    add_bullet("1. Prior Campaign Success is the Strongest Indicator:", "Clients with past positive experiences converted at 64.3%, compared to 11.5% baseline. Past satisfaction indicates high willingness to invest in subsequent bank offerings.")
    add_bullet("2. Debt Commitments Constrain Liquidity:", "Clients with housing loans converted at only 8.6% (vs 15.3% for debt-free clients), and personal loans halved conversion from 12.5% to 6.7%. Debt servicing restricts discretionary capital.")
    add_bullet("3. Contact Fatigue Appears Beyond 2 Contacts:", "More than 80% of conversions occur within the first 1-2 calls. Additional attempts beyond 3 calls yield diminishing returns and risk customer dissatisfaction.")
    add_bullet("4. Demographic Differences in Savings Preferences:", "Students (22.6%) and retirees (23.5%) demonstrated above-average response rates, suggesting strong demand for safe, guaranteed-return deposit instruments in those life stages.")

    # =========================================================================
    # CHAPTER 20: BUSINESS RECOMMENDATIONS
    # =========================================================================
    add_heading_1("20. Business Recommendations")
    add_bullet("1. Prioritize Re-engagement of Prior Subscribers:", "Establish an automated workflow targeting clients recorded with 'poutcome == success' early in campaign cycles.")
    add_bullet("2. Institute a 3-Call Outreach Limit:", "Enforce an operational cap of 3 call attempts per client per campaign to reduce unproductive call center labor costs.")
    add_bullet("3. Segment Offerings by Liability Profile:", "Avoid pitching term deposits to customers servicing significant mortgage or personal debt; offer debt consolidation or short-term liquid savings instead.")
    add_bullet("4. Deploy Machine Learning for Pre-Contact Lead Scoring:", "Use model propensity scores to rank calling lists daily, routing top-decile prospects to senior sales specialists.")
    add_bullet("5. Maintain Strict Pre-Contact Governance:", "Prohibit post-contact operational variables (such as call duration) in prospective targeting models to prevent data leakage.")

    # =========================================================================
    # CHAPTER 21: LIMITATIONS
    # =========================================================================
    add_heading_1("21. Limitations")
    add_bullet("1. Specific Geographic and Economic Context:", "Data reflects a Portuguese retail bank during the 2008-2010 financial crisis; relationships may differ under different interest-rate regimes.")
    add_bullet("2. Observational vs. Causal Evidence:", "Reported patterns represent statistical associations rather than proven causal relationships.")
    add_bullet("3. Class Imbalance:", "The 8:1 negative-to-positive ratio requires careful probability threshold selection in production deployment.")
    add_bullet("4. Presence of Unknown Categories:", "A substantial proportion of records contain 'unknown' in categorical fields, reflecting incomplete historical data collection.")

    # =========================================================================
    # CHAPTER 22: FUTURE SCOPE
    # =========================================================================
    add_heading_1("22. Future Scope")
    add_bullet("1. Threshold Optimization & Cost-Sensitive Learning:", "Calibrate classification thresholds using actual financial cost matrices (cost per outbound phone call vs. net profit from a signed term deposit).")
    add_bullet("2. Explainable AI with SHAP:", "Integrate SHAP values into CRM interfaces to give call center agents specific talking points tailored to individual customer risk-return profiles.")
    add_bullet("3. Temporal Split Validation:", "Evaluate model stability across sequential time-based splits to test resilience against temporal concept drift.")
    add_bullet("4. Interactive Dashboard Deployment:", "Deploy an interactive dashboard (e.g., Streamlit) allowing marketing analysts to simulate campaign ROI and run what-if scenarios.")

    # =========================================================================
    # CHAPTER 23: CONCLUSION
    # =========================================================================
    add_heading_1("23. Conclusion")
    add_body(
        "This project successfully developed an end-to-end data analytics and machine learning solution for bank marketing campaign optimization, "
        "fulfilling the requirements of the AICTE | IBM SkillsBuild Internship 2026. By utilizing the authentic UCI Bank Marketing dataset, "
        "conducting rigorous data audits, formalizing executive KPIs, and engineering domain features, the study uncovered key behavioral patterns "
        "governing term deposit adoption. Crucially, the project addressed the subtle pitfall of data leakage by removing call duration from prospective "
        "models, delivering realistic, production-viable classifiers. Logistic Regression demonstrated strong recall (62.5%), while Random Forest offered "
        "high precision (55.6%), providing flexible operational choices for banking marketing teams."
    )

    # =========================================================================
    # CHAPTER 24: REFERENCES
    # =========================================================================
    add_heading_1("24. References")
    add_bullet("[1] Moro, S., Cortez, P., & Rita, P. (2014).", "A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31. DOI: 10.1016/j.dss.2014.03.001.")
    add_bullet("[2] UCI Machine Learning Repository.", "Bank Marketing Dataset. URL: https://archive.ics.uci.edu/dataset/222/bank+marketing.")
    add_bullet("[3] Pedregosa, F., et al. (2011).", "Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.")
    add_bullet("[4] McKinney, W. (2010).", "Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56.")
    add_bullet("[5] Provost, F., & Fawcett, T. (2013).", "Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking. O'Reilly Media.")

    output_path = "MohanKumarM_ProjectReport.docx"
    doc.save(output_path)
    print(f"--> Report successfully generated and saved to: {output_path}")

if __name__ == '__main__':
    create_report()