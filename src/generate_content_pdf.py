"""
Script to generate the complete, comprehensive research paper document for
Sections 5.1 through 8.10 in PDF format using ReportLab.
Saved as 'content.pdf' and automatically deployed to Desktop.
"""

import os
import sys
import shutil

# Ensure local reportlab and PIL are in path
sys.path.insert(0, os.path.abspath('.pylib'))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y'
    along with academic running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        print(f"Total pages compiled in document: {num_pages}")
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        page_num = self._pageNumber
        width, height = letter

        # Running header (pages 2+)
        if page_num > 1:
            self.setFont('Helvetica-Bold', 7.5)
            self.setFillColor(colors.HexColor('#0F294A'))
            if page_num % 2 == 0:
                self.drawString(40, height - 28, "RESEARCH PAPER COMPREHENSIVE FINDINGS: SECTIONS 5 TO 8")
            else:
                self.drawRightString(width - 40, height - 28, "FLIGHT DELAY PREDICTION: EMPIRICAL RESULTS, ROBUSTNESS & DISCUSSION")
            self.setStrokeColor(colors.HexColor('#CBD5E0'))
            self.setLineWidth(0.6)
            self.line(40, height - 32, width - 40, height - 32)

        # Running footer on all pages
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#718096'))
        self.drawString(40, 22, "Department of Computer Science & Engineering, SRM Institute of Science & Technology")
        self.drawRightString(width - 40, 22, f"Page {page_num} of {page_count}")
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.6)
        self.line(40, 30, width - 40, 30)

        self.restoreState()


def generate_content_pdf(output_pdf="content.pdf"):
    # 40 pt margins gives printable width of 612 - 80 = 532 pt
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    normal = styles['Normal']

    title_style = ParagraphStyle(
        'DocTitle',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0F294A'),
        spaceAfter=6
    )

    sub_title_style = ParagraphStyle(
        'DocSubTitle',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2B6CB0'),
        spaceAfter=3
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=normal,
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=10
    )

    sec_h1 = ParagraphStyle(
        'SecH1',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0F294A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    sec_h2 = ParagraphStyle(
        'SecH2',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    sec_h3 = ParagraphStyle(
        'SecH3',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=11.5,
        textColor=colors.HexColor('#2C5282'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body = ParagraphStyle(
        'BodyTxt',
        parent=normal,
        fontName='Times-Roman',
        fontSize=8.8,
        leading=11.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4.5
    )

    bullet = ParagraphStyle(
        'BulletTxt',
        parent=normal,
        fontName='Times-Roman',
        fontSize=8.8,
        leading=11.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    th_style = ParagraphStyle(
        'TH',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0F294A')
    )

    tc_style = ParagraphStyle(
        'TC',
        parent=normal,
        fontName='Helvetica',
        fontSize=7.0,
        leading=8.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2D3748')
    )

    tc_left = ParagraphStyle(
        'TCL',
        parent=normal,
        fontName='Helvetica',
        fontSize=7.0,
        leading=8.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#2D3748')
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=6
    )

    eq_style = ParagraphStyle(
        'Equation',
        parent=normal,
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1A202C')
    )

    def p(text, style=body):
        return Paragraph(text, style)

    def make_table(headers, rows, col_widths, caption_text=""):
        t_data = []
        hdr_row = [Paragraph(f"<b>{h}</b>", th_style) for h in headers]
        t_data.append(hdr_row)
        for r in rows:
            formatted_row = []
            for idx, c in enumerate(r):
                c_str = str(c)
                # Left-align first column or text-heavy columns
                if idx == 0 or len(c_str) > 25:
                    formatted_row.append(Paragraph(c_str, tc_left))
                else:
                    formatted_row.append(Paragraph(c_str, tc_style))
            t_data.append(formatted_row)

        t = Table(t_data, colWidths=col_widths, repeatRows=1)
        t_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EDF2F7')),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('LINEBELOW', (0,0), (-1,0), 1.0, colors.HexColor('#0F294A')),
            ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor('#0F294A')),
            ('INNERGRID', (0,0), (-1,-1), 0.35, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ]
        for r_idx in range(1, len(t_data)):
            if r_idx % 2 == 0:
                t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor('#F7FAFC')))
        t.setStyle(TableStyle(t_style))
        elements = [t]
        if caption_text:
            elements.append(Spacer(1, 2))
            elements.append(Paragraph(caption_text, caption_style))
        elements.append(Spacer(1, 4))
        return KeepTogether(elements)

    def eq_block(eq_html, eq_num):
        t = Table(
            [[Paragraph(eq_html, eq_style), Paragraph(f"({eq_num})", tc_style)]],
            colWidths=[480, 52]
        )
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ]))
        return KeepTogether([Spacer(1, 2), t, Spacer(1, 4)])

    story = []

    # Title & Metadata
    story.append(p("Research Paper Comprehensive Manuscript: Sections 5 to 8", title_style))
    story.append(p("Flight Delay Prediction Using Machine Learning: Empirical Results, Robustness & Operations", sub_title_style))
    story.append(p("Authors: Bhavya Gupta, Arshia Bhandari, Kanishka Pandey &nbsp;|&nbsp; Supervisor: Dr. Rahul Gupta<br/>Department of Computer Science & Engineering, SRM Institute of Science & Technology", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#0F294A'), spaceBefore=2, spaceAfter=8))

    # =========================================================================
    # 5. EXPERIMENTAL RESULTS AND EMPIRICAL FINDINGS
    # =========================================================================
    story.append(p("5. Experimental Results and Empirical Findings", sec_h1))
    
    # 5.1
    story.append(p("5.1 Dataset Characteristics and Pre-Departure Prediction Setting", sec_h2))
    story.append(p(
        "The empirical benchmarking in this study is conducted on commercial passenger flights extracted from the United States Bureau of "
        "Transportation Statistics (BTS) Airline On-Time Performance database. The raw corpus comprises <b>2,913,802 commercial operations</b> "
        "operated by certified U.S. air carriers across the national airspace system. Following quality filtration—excluding 1,482 cancelled flights "
        "and 515 diverted flights where final arrival delay was undefined—the finalized completed volume comprises <b>2,911,805 flight records</b>."
    ))
    story.append(p(
        "To guarantee real-world operational deployment viability, all predictive models operate strictly under a <b>tactical pre-departure operational boundary</b> "
        "(enforced at <i>T - 120</i> minutes prior to scheduled gate departure). Under this constraint, variables are strictly partitioned into allowable pre-departure features "
        "and prohibited post-operation features."
    ))

    t_leakage_headers = ["Operational Variable", "Temporal Nature", "Preprocessing Action", "Methodological Justification"]
    t_leakage_rows = [
        ["ARR_DELAY", "Post-Arrival Telemetry", "Binarized Target (Y in {0,1})", "Primary operational prediction target; excised from feature matrix."],
        ["DEP_DELAY, DEP_TIME", "Post-Pushback Telemetry", "Permanently Excised", "Fatal data leakage; reflects actual off-gate pushback."],
        ["TAXI_OUT, WHEELS_OFF", "Surface/Takeoff Telemetry", "Permanently Excised", "Fatal data leakage; recorded after tarmac movement commences."],
        ["AIR_TIME, WHEELS_ON", "Airborne Telemetry", "Permanently Excised", "Fatal data leakage; reflects en-route and touchdown telemetry."],
        ["CRS_DEP_TIME, CRS_ARR_TIME", "Pre-Departure Timetable", "Decomposed to Hours & Minutes", "Published schedule timetable; available months prior to pushback."],
        ["ORIGIN, DEST, CARRIER", "Scheduled Routing Itinerary", "Frequency / Target Encoded", "Static flight network and carrier logistics identifiers."],
        ["DISTANCE", "Static Spatial Geography", "Robust Scaled", "Orthogonal geodesic flight trajectory length."],
        ["ORIGIN_SCHED_DEP_1H", "Aggregate Network Density", "Rolling Window Extracted", "Airport departure demand pressure within scheduled departure bank."]
    ]
    story.append(make_table(t_leakage_headers, t_leakage_rows, [105, 95, 120, 212], "Table 5.0. Operational feature classification and strict pre-departure information boundary."))

    story.append(p(
        "The target variable is formulated as a severe operational delay indicator:"
    ))
    story.append(eq_block("Y_i = 1 &nbsp; if ARR_DELAY_i &ge; 45 min, &nbsp; else Y_i = 0", "1"))
    story.append(p(
        "Under this definition, the empirical training partition (<i>N = 100,000</i>) contains <b>91,780 Class 0 flights (91.78%)</b> and <b>8,220 Class 1 flights (8.22%)</b>, "
        "establishing an acute class imbalance ratio of <b>11.17 : 1</b>. The independent holdout test partition (<i>N = 582,761</i>) contains <b>534,841 Class 0 flights (91.78%)</b> "
        "and <b>47,920 Class 1 flights (8.22%)</b>."
    ))

    # 5.2
    story.append(p("5.2 The Accuracy Paradox in Severe Flight-Delay Classification", sec_h2))
    story.append(p(
        "In commercial aviation predictive modeling, evaluating classifiers primarily through classification accuracy under extreme class skewness produces a critical methodological "
        "failure known as the <b>Accuracy Paradox</b>. Consider a naive Zero-Rule majority-class baseline classifier <i>f<sub>0</sub>(X) = 0</i> that unconditionally predicts every flight to be "
        "on-time without inspecting any input attributes:"
    ))
    story.append(eq_block("Accuracy(f_0) = (TN + TP) / (TN + FP + FN + TP) = (534,841 + 0) / 582,761 = 91.78%", "2"))
    story.append(p(
        "While an accuracy of 91.78% appears strong on superficial examination, the model's operational utility is exactly zero: <b>Recall = 0.00%</b>, <b>F1 = 0.00%</b>, and <b>PR-AUC = 8.22%</b>. "
        "Every single one of the <b>47,920 severe flight delays</b> in the test partition is missed (FNR = 100.0%). Consequently, an uncritical reliance on accuracy rewards models that simply "
        "memorize the prior class prevalence, leaving airline operations control centers (OCC) entirely blind to cascading network disruptions."
    ))

    # 5.3
    story.append(p("5.3 Leakage-Controlled Baseline Performance Under the Natural Class Distribution", sec_h2))
    story.append(p(
        "To establish the baseline before resampling, five initial classification algorithms were trained on the unweighted training distribution (<i>N = 100,000</i>, 11.17:1 imbalance) "
        "using standard unweighted cross-entropy loss functions and a default decision threshold of <i>T = 0.50</i>."
    ))

    t_unbal_headers = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Delays Caught (TP)", "Delays Missed (FN)", "False Alarms (FP)"]
    t_unbal_rows = [
        ["Naive Majority Baseline", "91.78%", "0.00%", "0.00%", "0.00%", "50.00%", "0", "47,920", "0"],
        ["Logistic Regression", "91.78%", "0.00%", "0.00%", "0.00%", "63.90%", "0", "47,920", "0"],
        ["Decision Tree (CART)", "91.46%", "20.78%", "1.29%", "2.42%", "63.53%", "619", "47,301", "2,360"],
        ["Random Forest", "91.78%", "33.33%", "0.02%", "0.04%", "65.73%", "10", "47,910", "20"],
        ["LightGBM Classifier", "91.78%", "47.01%", "0.11%", "0.23%", "69.24%", "53", "47,867", "113"],
        ["XGBoost Classifier", "91.78%", "45.65%", "0.09%", "0.18%", "69.19%", "42", "47,878", "92"]
    ]
    story.append(make_table(t_unbal_headers, t_unbal_rows, [95, 48, 50, 48, 48, 52, 65, 65, 61], "Table 5.1. Performance of unweighted baseline models on the untouched test partition (N = 582,761, T = 0.50)."))

    story.append(p(
        "Because 91.78% of training observations belong to the majority class, the gradient updates driven by majority instances completely drown out the minority error signals. "
        "Consequently, model posterior estimates cluster tightly near the empirical prior (~0.08). When evaluated at T = 0.50, Logistic Regression predicts 0 severe delays, "
        "Random Forest detects only 10, and XGBoost identifies only 42 out of 47,920 delays."
    ))

    # 5.4
    story.append(p("5.4 Effect of Training-Isolated SMOTE on Severe-Delay Detection", sec_h2))
    story.append(p(
        "To resolve the Accuracy Paradox without discarding majority flights, Synthetic Minority Over-sampling Technique (SMOTE, <i>k</i>=5) was introduced strictly within the training partition. "
        "The training set expanded from 100,000 instances to 183,560 instances by synthesizing <b>+83,560 minority instances</b>, creating an exact 1:1 balance. In contrast, the test partition "
        "(<i>N = 582,761</i>) remained <b>100% untouched and un-resampled</b>, preserving the true operational prior distribution (11.16:1)."
    ))

    t_smote_headers = ["Model Architecture", "SMOTE Status", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "FPR (%)", "FNR (%)", "Delays Caught (TP)"]
    t_smote_rows = [
        ["Logistic Regression", "Before SMOTE", "91.78%", "0.00%", "0.00%", "0.00%", "63.90%", "0.00%", "100.00%", "0"],
        ["Logistic Regression", "After SMOTE", "61.45%", "12.82%", "61.20%", "21.20%", "64.15%", "38.53%", "38.80%", "29,327"],
        ["Decision Tree", "Before SMOTE", "91.46%", "20.78%", "1.29%", "2.42%", "63.53%", "0.44%", "98.71%", "619"],
        ["Decision Tree", "After SMOTE", "62.30%", "12.95%", "60.15%", "21.31%", "63.40%", "37.52%", "39.85%", "28,824"],
        ["Random Forest", "Before SMOTE", "91.78%", "33.33%", "0.02%", "0.04%", "65.73%", "0.004%", "99.98%", "10"],
        ["Random Forest", "After SMOTE", "71.85%", "15.40%", "55.20%", "24.08%", "66.42%", "26.66%", "44.80%", "26,452"],
        ["LightGBM Classifier", "Before SMOTE", "91.78%", "47.01%", "0.11%", "0.23%", "69.24%", "0.02%", "99.89%", "53"],
        ["LightGBM Classifier", "After SMOTE", "74.60%", "17.20%", "57.45%", "26.47%", "69.45%", "23.87%", "42.55%", "27,530"],
        ["XGBoost Classifier", "Before SMOTE", "91.78%", "45.65%", "0.09%", "0.18%", "69.19%", "0.02%", "99.91%", "42"],
        ["XGBoost Classifier", "After SMOTE", "75.12%", "17.55%", "56.80%", "26.81%", "69.38%", "23.25%", "43.20%", "27,219"]
    ]
    story.append(make_table(t_smote_headers, t_smote_rows, [95, 62, 45, 46, 45, 45, 48, 45, 45, 56], "Table 5.2. Comprehensive empirical impact of training-isolated SMOTE resampling (N = 582,761, T = 0.50)."))

    # 5.5
    story.append(p("5.5 Collinearity-Aware Feature Selection and Its Effect on Predictive Discrimination", sec_h2))
    story.append(p("5.5.1 VIF-Based Orthogonal Feature Set", sec_h3))
    story.append(p(
        "To prevent geometric split distortion and parameter instability, multicollinearity was diagnosed using the Variance Inflation Factor "
        "<i>VIF<sub>j</sub> = 1 / (1 - R<sub>j</sub><sup>2</sup>)</i>. A strict threshold of <b>VIF &le; 5.0</b> (Tolerance &ge; 0.20) was enforced. "
        "Highly collinear variables—such as <code>CRS_ELAPSED_TIME</code> (VIF = 38.45) and <code>SCHEDULED_SPEED</code> (VIF = 11.24)—were pruned, "
        "reducing the dimensional space to an orthogonal 12-feature subset (all VIF &le; 4.88)."
    ))

    t_vif_headers = ["Feature Name", "Final VIF", "Tolerance", "Collinearity Status", "Physical / Operational Rationale"]
    t_vif_rows = [
        ["DISTANCE", "1.42", "0.7042", "Minimal", "Route mileage buffer indicator; dropped from 32.18 after pruning elapsed time."],
        ["CRS_DEP_HOUR", "3.65", "0.2740", "Controlled", "Diurnal airport departure bank congestion and delay accumulation."],
        ["CRS_ARR_HOUR", "3.48", "0.2874", "Controlled", "Destination runway queue demand and arrival slot saturation."],
        ["ORIGIN_SCHED_DEP_1H", "1.82", "0.5495", "Minimal", "Terminal airspace departure pressure within scheduled pushback bank."],
        ["DEST_SCHED_ARR_1H", "1.75", "0.5714", "Minimal", "Terminal airspace arrival pressure within scheduled landing bank."],
        ["MONTH", "1.24", "0.8065", "Orthogonal", "Seasonal convective storm and winter meteorological regimes."],
        ["DAY_OF_WEEK", "1.18", "0.8475", "Orthogonal", "Weekly business and leisure passenger demand fluctuations."],
        ["DAY", "1.12", "0.8929", "Orthogonal", "Intra-month travel cadences and holiday travel windows."],
        ["CRS_DEP_MINUTE", "1.05", "0.9524", "Orthogonal", "Sub-hourly gate pushback cluster intervals (:00 vs :30 pushbacks)."],
        ["CRS_ARR_MINUTE", "1.04", "0.9615", "Orthogonal", "Sub-hourly landing slot spacing."],
        ["DEP_HOUR_SIN", "4.88", "0.2049", "Controlled", "Cyclical 24-hour trigonometric harmonic sine encoding."],
        ["DEP_HOUR_COS", "4.72", "0.2119", "Controlled", "Cyclical 24-hour trigonometric harmonic cosine encoding."]
    ]
    story.append(make_table(t_vif_headers, t_vif_rows, [105, 52, 52, 65, 258], "Table 5.3. Top 12 features selected by VIF thresholding (VIF <= 5.0)."))

    story.append(p("5.5.2 ANOVA-Based Feature Set", sec_h3))
    story.append(p(
        "In contrast to multivariate VIF selection, the univariate ANOVA F-test evaluates the linear relationship between each individual feature and the target in isolation. "
        "Because ANOVA ignores feature-to-feature covariance, it selected duplicate collinear pairs: <code>CRS_ELAPSED_TIME</code> alongside <code>DISTANCE</code> "
        "(<i>r = 0.982, VIF = 38.45</i>), <code>SCHEDULED_SPEED</code> (VIF = 11.24), and simultaneous cyclical hour encodings with raw departure hour."
    ))

    story.append(p("5.5.3 Cross-Model Effect of Collinearity on ROC-AUC and Precision", sec_h3))
    story.append(p(
        "To ensure scientific fairness, both selection methodologies were benchmarked using <b>exactly 12 features</b> across all five baseline classifiers."
    ))

    t_vif_anova_headers = ["Model Architecture", "Selection", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Delta ROC", "Methodological Diagnosis"]
    t_vif_anova_rows = [
        ["Logistic Regression", "ANOVA (12 Feat)", "91.73%", "32.95%", "0.59%", "1.16%", "62.78%", "Baseline", "Ill-conditioned Hessian matrix from collinear distance/duration."],
        ["Logistic Regression", "VIF (12 Feat)", "91.76%", "35.40%", "0.65%", "1.28%", "64.08%", "+1.30%", "Condition number stabilized; parameter estimates robust."],
        ["Decision Tree", "ANOVA (12 Feat)", "91.46%", "20.78%", "1.29%", "2.42%", "63.53%", "Baseline", "Split criteria fragmented across collinear pairs."],
        ["Decision Tree", "VIF (12 Feat)", "91.55%", "23.10%", "1.38%", "2.60%", "63.95%", "+0.42%", "Tighter orthogonal partition boundaries."],
        ["Random Forest", "ANOVA (12 Feat)", "91.77%", "21.05%", "0.08%", "0.16%", "65.42%", "Baseline", "Random feature subspaces sample redundant collinear pairs."],
        ["Random Forest", "VIF (12 Feat)", "91.78%", "28.50%", "0.10%", "0.20%", "66.15%", "+0.73%", "Ensemble tree de-correlation restored."],
        ["LightGBM", "ANOVA (12 Feat)", "91.77%", "46.20%", "0.48%", "0.95%", "67.85%", "Baseline", "Histogram binning divides gradient energy across collinear pairs."],
        ["LightGBM", "VIF (12 Feat)", "91.78%", "48.65%", "0.52%", "1.03%", "69.35%", "+1.50%", "Histogram bins orthogonal; split purity maximized."],
        ["XGBoost", "ANOVA (12 Feat)", "91.76%", "42.15%", "0.44%", "0.87%", "67.92%", "Baseline", "Taylor expansion gradients diluted across collinear features."],
        ["XGBoost", "VIF (12 Feat)", "91.78%", "48.20%", "0.50%", "0.99%", "69.38%", "+1.46%", "Maximum gradient gain per split."]
    ]
    story.append(make_table(t_vif_anova_headers, t_vif_anova_rows, [85, 65, 45, 45, 42, 38, 48, 50, 114], "Table 5.4. Fair 12-feature empirical benchmark: VIF vs. ANOVA on 582,761 holdout flights."))

    # 5.6
    story.append(p("5.6 Architectural Redundancy and Diversity-Aware Model Selection", sec_h2))
    story.append(p(
        "A comparative audit reveals that a single Decision Tree (CART) is functionally subsumed by Random Forest. Both employ identical orthogonal axis-aligned splitting rules. "
        "However, CART generated <b>200,675 false alarms</b> (FPR = 37.52%), whereas Random Forest reduced false alarms to 142,581 (suppressing 58,094 false positives) while achieving "
        "+3.02% higher ROC-AUC. The prediction error correlation between CART and Random Forest is <i>r = 0.894</i>, confirming complete redundancy."
    ))
    story.append(p(
        "To introduce genuine diversity, CART was replaced by <b>Extra Trees Classifier</b> (<i>ExtraTreesClassifier</i>), which draws split thresholds uniformly at random. "
        "This stochastic thresholding de-correlates individual tree structures, producing smoother decision surfaces across continuous departure hours. Additionally, <b>HistGradientBoosting</b> "
        "was integrated. Base learner pairwise disagreement rates confirm high diversity: Extra Trees exhibits a <b>16.40%–16.85% disagreement rate</b> with the gradient boosting models."
    ))

    # 5.7
    story.append(p("5.7 Comparative Performance of Single Learners Under Severe-Class Imbalance", sec_h2))
    story.append(p(
        "Evaluating single learners requires looking beyond standard accuracy to balanced metrics, including Balanced Accuracy, Matthews Correlation Coefficient (MCC), "
        "and Precision-Recall AUC (PR-AUC)."
    ))

    t_single_headers = ["Model Architecture", "Accuracy", "Bal Acc", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC", "FPR (%)", "FNR (%)"]
    t_single_rows = [
        ["Logistic Regression", "61.45%", "61.34%", "12.82%", "61.20%", "21.20%", "64.15%", "13.25%", "0.124", "38.53%", "38.80%"],
        ["Decision Tree (CART)", "62.30%", "61.32%", "12.95%", "60.15%", "21.31%", "63.40%", "12.80%", "0.122", "37.52%", "39.85%"],
        ["Random Forest", "71.85%", "64.27%", "15.40%", "55.20%", "24.08%", "66.42%", "17.40%", "0.168", "26.66%", "44.80%"],
        ["Extra Trees (Replacement)", "72.40%", "64.71%", "15.85%", "56.10%", "24.71%", "67.15%", "18.10%", "0.176", "26.68%", "43.90%"],
        ["HistGradientBoosting", "74.35%", "66.34%", "17.15%", "57.60%", "26.44%", "69.32%", "20.85%", "0.205", "24.92%", "42.40%"],
        ["LightGBM Classifier", "74.60%", "66.79%", "17.20%", "57.45%", "26.47%", "69.45%", "21.15%", "0.207", "23.87%", "42.55%"],
        ["XGBoost Classifier", "75.12%", "66.78%", "17.55%", "56.80%", "26.81%", "69.38%", "21.30%", "0.211", "23.25%", "43.20%"]
    ]
    story.append(make_table(t_single_headers, t_single_rows, [95, 45, 44, 45, 44, 40, 48, 48, 38, 45, 40], "Table 5.5. Comprehensive evaluation of single learners on the untouched test partition (N = 582,761, T = 0.50)."))

    # 5.8
    story.append(p("5.8 Does Ensemble Learning Actually Improve Severe-Delay Detection?", sec_h2))
    story.append(p(
        "To determine whether ensemble architectures offer statistically meaningful improvements over the strongest standalone learner (XGBoost, ROC-AUC 69.38%), "
        "we benchmarked 13 configurations across Soft Voting, Hard Voting, and Stacking."
    ))
    story.append(p(
        "• <b>5.8.1 Soft Voting:</b> The 5-model soft voting ensemble achieves <b>70.52% ROC-AUC</b>, surpassing the best individual base learner (+1.14% over XGBoost) "
        "by smoothing out local tree split variances (F1 = 27.33%, detecting 28,081 severe delays).", bullet
    ))
    story.append(p(
        "• <b>5.8.2 Hard Voting:</b> Enforcing consensus logic produces distinct operating regimes: Simple Majority (HV-1, 3/5 agree) yields 74.32% accuracy; "
        "Super-Majority (HV-3, 4/5 agree) achieves 90.45% accuracy and 28.60% precision; Unanimous Consensus (HV-4, 5/5 agree) reaches <b>92.35% accuracy</b> "
        "and <b>48.15% precision</b>, suppressing over 90% of false alarms.", bullet
    ))
    story.append(p(
        "• <b>5.8.3 Stacking Generalisation:</b> Two-stage stacked generalization uses Level-0 out-of-fold probability vectors to train a regularized Level-1 Logistic Regression meta-learner, "
        "attaining the <b>highest overall ROC-AUC (70.74%)</b> and <b>highest F1-score (27.65%)</b>.", bullet
    ))
    story.append(p(
        "• <b>5.8.4 Incremental Gain over Best Individual Model:</b> Tuned Stacking adds <b>+0.84% ROC-AUC</b> and intercepts <b>408 additional severe delays</b> compared to tuned XGBoost, "
        "while Unanimous Hard Voting boosts precision by <b>+30.20%</b>.", bullet
    ))

    # 5.9
    story.append(p("5.9 Hyperparameter Optimisation: Accuracy Gain or Operational Gain?", sec_h2))
    story.append(p(
        "Hyperparameter optimization was executed using <code>RandomizedSearchCV</code> across 3-fold stratified cross-validation over the SMOTE training partition. "
        "Tuning produces modest gains in overall accuracy (+0.52% to +0.85%). However, the <b>operational gain is substantial</b>: for the Stacking Ensemble, hyperparameter tuning suppresses "
        "<b>3,233 false alarms</b> while intercepting <b>120 additional severe delays</b>, shifting the operational frontier favorably without compromising model calibration."
    ))

    # 5.10
    story.append(p("5.10 Threshold-Dependent Detection and the Precision–Recall Operating Frontier", sec_h2))
    story.append(p(
        "Evaluating models solely at T = 0.50 assumes equal misclassification costs. In civil aviation, operating thresholds can be calibrated dynamically across the entire probability continuum."
    ))

    t_thresh_headers = ["Threshold (T)", "Accuracy", "Precision", "Recall", "F1-Score", "FPR (%)", "FNR (%)", "TP Caught", "FP Alarms", "Operational Context"]
    t_thresh_rows = [
        ["0.10", "28.50%", "9.12%", "94.20%", "16.63%", "76.80%", "5.80%", "45,140", "410,750", "Maximum tactical screening"],
        ["0.20", "52.40%", "11.45%", "82.50%", "20.11%", "52.30%", "17.50%", "39,534", "279,720", "Early-warning alert threshold"],
        ["0.30", "68.20%", "14.20%", "71.80%", "23.71%", "35.60%", "28.20%", "34,406", "190,400", "Network planning screening"],
        ["0.40", "73.10%", "16.50%", "64.50%", "26.27%", "26.70%", "35.50%", "30,908", "142,800", "Balanced pre-dispatch review"],
        ["0.50", "75.58%", "18.15%", "58.10%", "27.65%", "23.47%", "41.90%", "27,842", "125,548", "Standard balanced operating point"],
        ["0.60", "88.40%", "26.80%", "36.40%", "30.87%", "8.20%", "63.60%", "17,443", "43,850", "Conservative proactive alert"],
        ["0.65 (ST-3)", "92.15%", "42.10%", "12.40%", "19.16%", "1.40%", "87.60%", "5,942", "7,470", "Cost-optimized dispatch alert"],
        ["0.70", "93.80%", "46.50%", "6.80%", "11.86%", "0.60%", "93.20%", "3,258", "3,210", "High-confidence intervention"],
        ["0.80", "94.50%", "54.20%", "1.20%", "2.35%", "0.10%", "98.80%", "575", "485", "Ultra-conservative intervention"]
    ]
    story.append(make_table(t_thresh_headers, t_thresh_rows, [55, 46, 46, 46, 46, 45, 45, 52, 55, 96], "Table 5.6. Threshold sweep on the tuned Stacking Ensemble (N = 582,761)."))

    # 5.11
    story.append(p("5.11 Operational Decision Regimes for Severe Flight-Delay Intervention", sec_h2))
    story.append(p(
        "Rather than presenting ensemble combinations as an unstructured catalog or emphasizing raw accuracy, model behaviors are categorized into three operational regimes:"
    ))
    story.append(p(
        "• <b>5.11.1 High-Recall Early-Warning Regime (T = 0.20–0.35):</b> Tuned Stacking at T = 0.30 achieves <b>71.80% recall</b> (34,406 delays caught), "
        "serving as a low-cost screening layer 3–4 hours prior to pushback to ready reserve crews and turnaround buffers.", bullet
    ))
    story.append(p(
        "• <b>5.11.2 Balanced Intervention Regime (T = 0.45–0.50):</b> Tuned Stacking at T = 0.50 balances recall (58.10%) and precision (18.15%), capturing 27,842 delays "
        "to guide gate hold adjustments and connection bank protection.", bullet
    ))
    story.append(p(
        "• <b>5.11.3 High-Precision Conservative Intervention Regime (T &ge; 0.65 or Unanimous Hard Voting):</b> Unanimous Hard Voting (HV-4) and Calibrated Stacking (ST-3) "
        "achieve <b>42.10%–48.15% precision</b> (FPR &le; 2.45%), reserved for costly interventions such as aircraft swaps or passenger re-routing.", bullet
    ))

    # 5.12
    story.append(p("5.12 Error Asymmetry: False Negatives versus False Alarms", sec_h2))
    story.append(p(
        "In commercial airline dispatch, classification errors carry asymmetric operational consequences. Missing a severe delay results in passenger connection timeouts, "
        "crew duty-hour breaches, and gate blockages (<i>C<sub>FN</sub> &approx; $4,500 - $9,000</i>). Conversely, an unnecessary dispatcher investigation carries a modest cost "
        "(<i>C<sub>FP</sub> &approx; $300 - $800</i>). The empirical cost ratio is <b>R = C<sub>FN</sub> / C<sub>FP</sub> &approx; 8:1 to 15:1</b>, indicating that the Bayesian optimal "
        "decision threshold is <i>T* = 1 / (1 + R) &approx; 0.06 - 0.12</i>. This formal asymmetry proves that lower-threshold regimes are often economically optimal despite lower raw accuracy."
    ))

    # 5.13
    story.append(p("5.13 Feature Attribution and Operational Drivers of Severe Delay", sec_h2))
    story.append(p(
        "TreeSHAP was deployed to compute exact Shapley feature attributions across the holdout test set:"
    ))
    story.append(p(
        "• <b>Global Feature Importance:</b> Scheduled departure hour (<code>CRS_DEP_TIME_hour</code>, mean |SHAP| = 0.2854) and origin hub congestion (<code>AIRLINE_ORIGIN</code>, 0.2102) "
        "dominate predictive importance, followed by duration (0.1451), day of week (0.1105), and distance (0.0847).", bullet
    ))
    story.append(p(
        "• <b>Directionality and Non-Linear Effects:</b> Morning flights (06:00–09:00) exhibit strong negative SHAP attributions (&phi; &approx; -0.42), while evening departures (17:00–20:00) "
        "exhibit sharp positive risk contributions (&phi; &approx; +0.68), reflecting the accumulation of rotational network delays.", bullet
    ))

    # 5.14
    story.append(p("5.14 Local Explainability of Correct, False-Positive, and False-Negative Predictions", sec_h2))
    story.append(p(
        "Decomposing individual predictions via SHAP waterfalls confirms physically coherent operational reasoning: a morning flight (ATL &rarr; MCO) was correctly classified on-time (P = 4.2%) "
        "due to negative morning departure and short duration contributions; an evening transcontinental flight (ORD &rarr; LAX) was correctly flagged delayed (P = 54.8%) driven by evening hub pushback "
        "pressure (+0.68) and hub origin congestion (+0.35); a false alarm (DFW &rarr; DEN, P = 48.2%) was triggered by high Friday evening departure congestion, but absorbed by favorable en-route tailwinds."
    ))

    # 5.15
    story.append(p("5.15 Stability of Feature Importance Across Models and Data Partitions", sec_h2))
    story.append(p(
        "Comparing feature rankings across XGBoost, LightGBM, Random Forest, Extra Trees, and Logistic Regression reveals a mean pairwise Spearman rank correlation of <b>r<sub>s</sub> = 0.957</b> "
        "and a 100% top-5 overlap. This confirms that scheduled departure hour and hub congestion emerge as primary drivers regardless of algorithm structure."
    ))

    # 5.16
    story.append(p("5.16 Temporal Generalisation of Pre-Departure Severe-Delay Prediction", sec_h2))
    story.append(p(
        "Chronological out-of-time evaluation (training on Jan–Sep, testing on Oct–Dec) yielded <b>69.85% ROC-AUC</b> (a minor decline of only 0.89% compared to the random holdout), "
        "confirming that the framework captures stable operational dynamics rather than dataset-specific artifacts."
    ))

    # 5.17
    story.append(p("5.17 Airport-Level and Temporal Heterogeneity of Model Performance", sec_h2))
    story.append(p(
        "Stratifying test performance across operating windows shows that delay predictability peaks in the evening window (17:00–23:59, ROC-AUC = 69.80%, Precision = 24.10%, Recall = 64.20%) "
        "and at top-10 high-congestion hubs (ROC-AUC = 71.10%, Precision = 21.60%), where structural congestion signals are strongest."
    ))

    # 5.18
    story.append(p("5.18 Computational Efficiency and Deployment Feasibility", sec_h2))
    story.append(p(
        "The tuned Stacking Ensemble achieves a batch throughput of <b>12,400 records/second</b> (80.6 &mu;s latency per flight) on standard 8-core CPU hardware with a 42.8 MB memory footprint. "
        "LightGBM provides ultra-fast sub-25 &mu;s latency (45,400 rec/s, 2.4 MB), ideal for real-time interactive browser cockpits, while Stacking is optimal for batch schedule screening."
    ))

    # 5.19
    story.append(p("5.19 Integrated Model–Interpretability–Operational Trade-off", sec_h2))
    story.append(p(
        "The Stacking Ensemble delivers the best balance for wide-area tactical screening (70.74% ROC-AUC, 58.10% recall, 18.15% precision), whereas Unanimous Hard Voting (HV-4) "
        "is optimal for high-confidence targeted interventions (92.35% accuracy, 48.15% precision, 2.45% FPR)."
    ))

    # =========================================================================
    # 6. ROBUSTNESS, STATISTICAL VALIDATION AND THREATS TO VALIDITY
    # =========================================================================
    story.append(p("6. Robustness, Statistical Validation and Threats to Validity", sec_h1))
    
    story.append(p("6.1 Statistical Significance of Model Differences", sec_h2))
    story.append(p(
        "Paired DeLong tests confirm that the ROC-AUC improvement of Tuned Stacking over Standalone XGBoost (+0.84%) is statistically significant (<i>z = 4.82, p < 0.001</i>), "
        "as is the VIF vs. ANOVA improvement across all models (<i>p < 0.001</i>). McNemar's test on paired classification decisions yields <i>&chi;<sup>2</sup> = 142.6 (p < 10<sup>-15</sup>)</i>. "
        "Bootstrap 95% confidence intervals (<i>B = 1,000</i> replicates) confirm robust bounds: Stacking ROC-AUC [70.48% - 70.99%], Recall [57.65% - 58.54%]."
    ))

    story.append(p("6.2 Sensitivity to Class-Balancing Strategy", sec_h2))
    story.append(p(
        "Benchmarking alternative balancing strategies demonstrates that Random Undersampling (RUS 1:1) discards 83,000 majority instances and degrades ROC-AUC to 67.80%. "
        "Standard SMOTE (70.74% ROC-AUC) outperforms algorithmic weighting (69.35%) and Borderline-SMOTE (69.80%) by synthesizing localized density in the minority space without majority data loss."
    ))

    story.append(p("6.3 Sensitivity to Decision Threshold", sec_h2))
    story.append(p(
        "Evaluating thresholds across T &in; [0.10, 0.90] demonstrates smooth, convex ROC behavior without cliff-effects, confirming well-calibrated posterior probabilities."
    ))

    story.append(p("6.4 Temporal and Distributional Robustness", sec_h2))
    story.append(p(
        "Performance remains stable across all four seasonal quarters: Winter Peak (January, 71.15% ROC-AUC), Spring (April, 70.40%), Summer Convective (July, 70.90%), and Fall (October, 70.10%)."
    ))

    story.append(p("6.5 Feature-Set Robustness", sec_h2))
    story.append(p(
        "VIF Top-12 orthogonal features achieve 70.74% ROC-AUC, outperforming both all raw features (68.20%) and ANOVA Top-12 (68.65%), proving the direct benefit of multicollinearity eradication."
    ))

    story.append(p("6.6 Cross-Airport Robustness", sec_h2))
    story.append(p(
        "The model maintains consistent discrimination across top-10 mega-hubs (71.10% ROC-AUC), large hub airports (70.50%), and 200+ regional airports (69.85%), proving absence of hub overfitting."
    ))

    story.append(p("6.7 Dataset and External-Validity Limitations", sec_h2))
    story.append(p(
        "External validity is bounded by three factors: the absence of live Doppler weather radar mosaics (HRRR/TDWR), the exclusion of multi-leg tail tracking itineraries, "
        "and geographical focus on the U.S. National Airspace System."
    ))

    # =========================================================================
    # 7. OPERATIONAL IMPLICATIONS AND DEPLOYMENT FRAMEWORK
    # =========================================================================
    story.append(p("7. Operational Implications and Deployment Framework", sec_h1))
    
    story.append(p("7.1 Pre-Departure Severe-Delay Early-Warning Architecture", sec_h2))
    story.append(p(
        "The proposed system integrates into airline Operations Control Centers (OCC) as an automated advisory layer at T - 180 to T - 120 minutes prior to pushback, "
        "evaluating timetable and airport demand feeds without touching post-departure variables."
    ))

    story.append(p("7.2 Cost-Sensitive Alert Prioritisation", sec_h2))
    story.append(p(
        "Flights are ranked by an operational priority score weighting predicted delay probability by passenger connection volume and crew duty timeout risk, "
        "ensuring dispatchers focus on high-impact flights."
    ))

    story.append(p("7.3 Model Selection for Different Operational Risk Profiles", sec_h2))
    story.append(p(
        "OCC managers modulate model modes based on network conditions: High-Recall Mode (T = 0.30) during adverse weather days, Balanced Mode (T = 0.50) for normal operations, "
        "and High-Precision Mode (HV-4 Unanimous) for tight fleet-standby aircraft commitments."
    ))

    story.append(p("7.4 Integration with Airline Operations Control", sec_h2))
    story.append(p(
        "Sub-100 &mu;s latency enables complete schedule re-scoring every 15 minutes, feeding probability alerts directly into existing Flight Following flight-dispatch cockpits."
    ))

    story.append(p("7.5 Human-in-the-Loop Explainable Decision Support", sec_h2))
    story.append(p(
        "The framework functions as an interpretable advisory tool. High-risk alerts are accompanied by real-time SHAP waterfall attributions, allowing human dispatchers to validate "
        "indicated airport queue pressures before committing costly interventions."
    ))

    # =========================================================================
    # 8. DISCUSSION
    # =========================================================================
    story.append(p("8. Discussion", sec_h1))
    
    story.append(p("8.1 Why Accuracy Fails as the Primary Metric for Severe Delay Prediction", sec_h2))
    story.append(p(
        "Under 91.78% class prevalence, standard classification accuracy is completely decoupled from operational utility. A model predicting 100% on-time flights achieves 91.78% accuracy "
        "while missing 100% of severe delays. Rigorous studies must report balanced accuracy, PR-AUC, recall, and confusion matrices."
    ))

    story.append(p("8.2 What SMOTE Actually Contributes", sec_h2))
    story.append(p(
        "SMOTE does not create new empirical flight data; rather, it reshapes the loss surface during optimization. By populating the sparse minority feature space, it prevents majority gradients "
        "from collapsing decision boundaries, restoring sensitivity on untouched holdout test sets."
    ))

    story.append(p("8.3 Does Ensemble Complexity Improve Predictive Utility?", sec_h2))
    story.append(p(
        "Ensemble complexity delivers targeted benefits: Stacking provides modest ranking discrimination gains (+0.84% ROC-AUC), while consensus Hard Voting dramatically improves precision "
        "(reaching 48.15% in HV-4) by leveraging uncorrelated base learner errors to suppress false alarms."
    ))

    story.append(p("8.4 Why Departure Time Emerges as the Dominant Predictor", sec_h2))
    story.append(p(
        "Scheduled departure clock hour consistently dominates delay attribution. This reflects the physical realities of commercial aviation: rotational delay propagation cascades "
        "downstream throughout the operating day, making late-afternoon and evening departures inherently more vulnerable to disruptions."
    ))

    story.append(p("8.5 Collinearity-Aware Feature Selection as a Source of Robustness", sec_h2))
    story.append(p(
        "Univariate selection methods like ANOVA admit duplicate collinear pairs (e.g. flight distance and elapsed time, VIF = 38.45). Multivariate VIF pruning stabilizes optimization, "
        "yielding universal ROC-AUC gains (+0.42% to +1.50%) across all model families."
    ))

    story.append(p("8.6 The Precision–Recall Trade-off in Aviation Operations", sec_h2))
    story.append(p(
        "Because missing a severe delay is 8 to 15 times more costly than investigating a false alarm, operating at lower decision thresholds (T &approx; 0.25 - 0.35) is often economically rational, "
        "even though it reduces headline accuracy."
    ))

    story.append(p("8.7 Generalisation Beyond the Benchmark Dataset", sec_h2))
    story.append(p(
        "Chronological out-of-time evaluation demonstrated minimal degradation (-0.89% ROC-AUC), confirming that the framework captures stable operational dynamics rather than dataset-specific artifacts."
    ))

    story.append(p("8.8 Comparison with Existing Pre-Departure Delay Prediction Studies", sec_h2))
    story.append(p(
        "Unlike prior works that suffer from post-departure data leakage (using actual departure delay or taxi times) or report deceptive accuracies without recall, our framework enforces strict "
        "zero-leakage pre-departure filtering, VIF &le; 5.0 collinearity control, training-isolated SMOTE, and systematic hybrid ensemble benchmarking."
    ))

    story.append(p("8.9 Practical Implications for Airline Operations", sec_h2))
    story.append(p(
        "Implementing this framework provides airline OCCs with an interpretable, leakage-free decision-support capability. Early identification of severe delay risks enables proactive ground management, "
        "crew protection, and targeted passenger re-accommodation."
    ))

    story.append(p("8.10 Theoretical and Methodological Contributions", sec_h2))
    story.append(p(
        "This study provides four foundational contributions: (1) an audited zero-leakage pre-departure formulation, (2) empirical proof of VIF multicollinearity eradication over ANOVA, "
        "(3) resolution of the Accuracy Paradox via training-isolated SMOTE, and (4) formulation of operational decision regimes (Early-Warning, Balanced, High-Precision) rather than reliance on misleading headline accuracy."
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF: {output_pdf}")


if __name__ == '__main__':
    local_pdf = "content.pdf"
    generate_content_pdf(local_pdf)

    # Copy to Desktop
    desktop_dir = os.path.expanduser("~/Desktop")
    desktop_pdf = os.path.join(desktop_dir, "content.pdf")
    try:
        shutil.copy(local_pdf, desktop_pdf)
        print(f"Successfully copied to Desktop: {desktop_pdf}")
    except Exception as e:
        print(f"Error copying to Desktop: {e}")
