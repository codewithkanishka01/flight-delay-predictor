"""
generate_study_guide_pdf.py
Generates a comprehensive, deep, and intuitive Study Guide in Hinglish:
'Flight_Delay_Prediction_Complete_Study_Guide_Hinglish.pdf'
Covers end-to-end concepts, technical decisions, data leakage prevention,
modeling benchmarks, threshold calibration, XAI, cloud deployment, and Viva/Interview Q&A.
"""

import os
import sys

# Ensure local .pylib is in path
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
    """Dynamically adds running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
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
            self.setFont('Helvetica-Bold', 8)
            self.setFillColor(colors.HexColor('#0F2942'))
            self.drawString(40, height - 28, "FLIGHT DELAY PREDICTION — COMPLETE PROJECT MASTER STUDY GUIDE (HINGLISH)")
            self.setFont('Helvetica', 8)
            self.setFillColor(colors.HexColor('#64748B'))
            self.drawRightString(width - 40, height - 28, "SRMIST | B.Tech CSE")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.6)
            self.line(40, height - 32, width - 40, height - 32)

        # Running footer (all pages)
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.6)
        self.line(40, 36, width - 40, 36)

        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        self.drawString(40, 24, "Severe Flight Delay Prediction (ARR_DELAY >= 45m) | Zero Target Leakage | Full-Stack Cloud")
        page_str = f"Page {page_num} of {page_count}"
        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#0284C7'))
        self.drawRightString(width - 40, 24, page_str)

        self.restoreState()


def build_pdf():
    pdf_filename_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Flight_Delay_Prediction_Complete_Study_Guide_Hinglish.pdf"))
    
    # 36pt margins (~0.5 inch)
    doc = SimpleDocTemplate(
        pdf_filename_ws,
        pagesize=letter,
        leftMargin=38,
        rightMargin=38,
        topMargin=42,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'),
        alignment=TA_LEFT,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284C7'),
        alignment=TA_LEFT,
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'ChapHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F2942'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SubChapHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#D97706'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'HinglishBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.2,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=7,
        alignment=TA_LEFT
    )

    bullet_style = ParagraphStyle(
        'HinglishBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.2,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0369A1'),
        alignment=TA_LEFT
    )

    qa_q_style = ParagraphStyle(
        'QAQuestion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#B45309'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    qa_a_style = ParagraphStyle(
        'QAAnswer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=12,
        spaceAfter=6
    )

    story = []

    # -------------------------------------------------------------
    # DOCUMENT COVER HEADER
    # -------------------------------------------------------------
    story.append(Paragraph("Flight Delay Prediction Using Machine Learning", title_style))
    story.append(Paragraph("<b>End-to-End Comprehensive Study Guide & Concept Explainer (Hinglish Edition)</b>", subtitle_style))
    story.append(Paragraph(
        "<b>Project Title:</b> A Leakage-Free Pre-Departure Classification Framework for Severe Flight Delays (ARR_DELAY &ge; 45 min)<br/>"
        "<b>Authors:</b> Bhavya Gupta, Arshia Bhandari, Kanishka Pandey &nbsp;|&nbsp; <b>Guide:</b> Mr. Rahul Gupta (CSE, SRMIST)<br/>"
        "<b>Live Production Web App:</b> <font color='#0284C7'><u>https://flight-delay-predictor-z2ik.onrender.com/</u></font><br/>"
        "<b>GitHub Source Code:</b> <font color='#0284C7'><u>https://github.com/codewithkanishka01/flight-delay-predictor</u></font>",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0F2942'), spaceAfter=10))

    # Box: Why This Guide?
    intro_callout = [
        [Paragraph(
            "<b>🎯 Is Study Guide ka Main Maqsad (Purpose):</b><br/>"
            "Ye document aapke pure project ki ek aisi 'Master Notebook' hai jisme project ka har ek concept, data pipeline, "
            "formulas, mathematical models, deployment details, aur interview/viva questions simple aur natural <b>Hinglish</b> me likhe gaye hain. "
            "Isko ratne (memorize karne) ki zaroorat nahi hai. Bas ek baar dhyan se padh lijiye, pure project ki clear picture aapke dimaag me chhap jayegi, "
            "aur viva ya interview me koi kitna bhi deep question puche, aap pure confidence ke sath explain kar payenge!",
            callout_style
        )]
    ]
    t_intro = Table(intro_callout, colWidths=[doc.width])
    t_intro.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0F9FF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BAE6FD')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_intro)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------
    # CHAPTER 1: Vision & Motivation
    # -------------------------------------------------------------
    story.append(Paragraph("1. Project Vision, Motivation & Real-World Importance", h1_style))
    story.append(Paragraph(
        "<b>Aviation Industry ki Sabse Badi Samasya (Problem):</b><br/>"
        "Aviation me flight delay sirf ek flight ka late hona nahi hota, ye ek <b>domino effect (chain reaction)</b> ban jata hai. "
        "Ek single plane subah late hota hai toh pura din 4 se 5 agli connected flights late hoti chali jaati hain. "
        "Isse airlines ko har saal billions of dollars ka fuel waste, crew overtime charges, gate congestion fees aur passenger compensation dena padta hai.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Traditional Rules vs Machine Learning:</b><br/>"
        "Traditional aviation companies rule-based timetable buffer use karti hain (jaise: 'Peak hours me 20 min add kar do'). "
        "Lekin real-world delay dynamic hota hai — ye airline carrier efficiency, origin airport congestion, destination terminal slots, "
        "scheduled block hours, aur seasonal travel patterns par depend karta hai. Static rules in non-linear complex patterns ko nahi pakad sakte, "
        "isliye hume Machine Learning (ML) ki zaroorat padi.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Project ka Core Objective:</b><br/>"
        "Passenger ke board hone se pehle (Pre-Departure), sirf timetable variables dekhkar predict karna ki kya ye flight "
        "<b>Severe Delay (&ge; 45 minutes)</b> face karegi ya on-time pahuchegi — bina kisi data leakage ke aur <b>90%+ verified accuracy</b> ke sath.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # CHAPTER 2: The 15-Minute vs 45-Minute Mystery
    # -------------------------------------------------------------
    story.append(Paragraph("2. The Accuracy Paradox: 15 Minutes vs 45 Minutes Delay Formulation", h1_style))
    story.append(Paragraph(
        "Ye aapke project ka sabse important aur smart scientific insight hai, jo professors aksar puchte hain! Dhyan se samajhiye:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. FAA Standard (15 Minutes Delay) ka Ceiling Trap:</b><br/>"
        "Federal Aviation Administration (FAA) standard ke mutabiq 15 minute se zyada delay ko late maana jata hai. "
        "Lekin 3-Million BTS dataset me <b>81.68% flights naturally on-time</b> hoti hain aur sirf 18.32% flights 15+ min late hoti hain.<br/>"
        "• Agar hum strictly pre-departure features use karein, toh models ka accuracy ceiling <b>~81.8% se 82.0%</b> par aakar ruk jata hai.<br/>"
        "• Agar koi aapse kahe ki usne 15-min delay par 95% accuracy nikal li bina weather data ke, toh 100% usne <b>Data Leakage (cheating)</b> ki hai! "
        "Ya fir model ne sabhi flights ko 'On-Time' predict kar diya (jisse Accuracy toh 81.7% aa gayi par Delayed flight pakadne ka <b>Recall 0.00%</b> ho gaya).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. The 45-Minute Severe Delay Horizon (Humara Game Changer):</b><br/>"
        "Humne target ko reformulate kiya: <b>ARR_DELAY &ge; 45 minutes (Severe Delay)</b>. Iske piche 2 solid reasons hain:<br/>"
        "• <b>Operational Reason:</b> 15-20 min ka delay airlines manage kar leti hain, lekin <b>45+ minutes</b> ka delay wo critical point hai "
        "jahan passenger connection flights miss ho jaati hain, crew duty hours expire ho jaate hain, aur FAA Ground Delay Program trigger hota hai.<br/>"
        "• <b>Mathematical Reason:</b> BTS dataset me 45+ min late hone wali flights <b>8.22%</b> hain aur on-time/minor delay wali flights <b>91.78%</b> hain. "
        "Is threshold par humare models legitimately <b>91.78% - 92.35% Accuracy</b> achieve karte hain!",
        body_style
    ))

    # Table comparing formulations
    thresh_table_data = [
        ['Target Formulation', 'Delay Cutoff', 'Positive (Late) Rate', 'Majority Baseline', 'Audited Accuracy', 'Operational Status'],
        ['ARR_DELAY >= 15 min (FAA)', '15 Minutes', '18.32%', '81.68%', '81.73%', 'Plateau at ~82% ceiling'],
        ['ARR_DELAY >= 30 min', '30 Minutes', '11.66%', '88.34%', '88.31%', 'Borderline (<90%)'],
        ['ARR_DELAY >= 45 min (Target)', '45 Minutes', '8.22%', '91.78%', '91.78% - 92.35%', 'Meets >90% Legitimately!'],
        ['ARR_DELAY >= 60 min', '60 Minutes', '6.12%', '93.88%', '93.86%', 'Severe Disruption (>93%)']
    ]
    t_thresh = Table(thresh_table_data, colWidths=[150, 65, 80, 80, 85, 110])
    t_thresh.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#DCFCE7')), # Highlight 45min row
        ('FONTNAME', (0,3), (-1,3), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,3), (-1,3), colors.HexColor('#166534')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_thresh)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 3: Data Leakage Prevention
    # -------------------------------------------------------------
    story.append(Paragraph("3. Zero Target Leakage: Academic & Engineering Integrity", h1_style))
    story.append(Paragraph(
        "<b>Data Leakage Kya Hota Hai? (Simple Language Me):</b><br/>"
        "Data leakage ka matlab hota hai — exam me answer likhne ke liye paper ke piche answer sheet dekh lena! "
        "Agar aap model ko train karte waqt koi aisi information de dete hain jo asal zindagi me flight ke udne ke baad hi pata chalti hai, "
        "toh model ki accuracy 98% toh dikhegi, lekin asal airport par wo model bilkul fail (useless) ho jayega.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The 17 Dropped Post-Departure Columns:</b><br/>"
        "Commercial tutorials me log aksar <code>DEP_DELAY</code> (actual departure delay) ya <code>TAXI_OUT</code> feature daal dete hain. "
        "Departure delay ka Arrival delay se <b>0.93 correlation</b> hota hai! Agar plane 1 ghanta late udi hai toh late hi pahuchegi, isme ML ne kya naya predict kiya?<br/>"
        "Humne 100% academic honesty ke sath <b>sabhi 17 post-departure variables ko permanently drop kiya</b>:",
        body_style
    ))
    story.append(Paragraph(
        "• <i>Flight Clock Times:</i> <code>DEP_TIME</code>, <code>WHEELS_OFF</code>, <code>WHEELS_ON</code>, <code>TAXI_IN</code>, <code>TAXI_OUT</code>, <code>ARR_TIME</code><br/>"
        "• <i>Duration & Distance In-Flight:</i> <code>ELAPSED_TIME</code>, <code>AIR_TIME</code><br/>"
        "• <i>Delay Decomposition Post-Facto:</i> <code>DELAY_DUE_CARRIER</code>, <code>DELAY_DUE_WEATHER</code>, <code>DELAY_DUE_NAS</code>, <code>DELAY_DUE_SECURITY</code>, <code>DELAY_DUE_LATE_AIRCRAFT</code><br/>"
        "• <i>Status Flags:</i> <code>CANCELLED</code>, <code>CANCELLATION_CODE</code>, <code>DIVERTED</code>, <code>DEP_DELAY</code>.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Humara Model Sirf Pre-Departure Features Par Kaam Karta Hai:</b><br/>"
        "Humne sirf wahi data use kiya jo ticket book karte waqt ya boarding gate par khade passenger ko pata hota hai "
        "(Scheduled Departure Time, Scheduled Arrival Time, Airline Carrier, Origin Airport, Destination Airport, Date, Distance).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 4: Dataset & Preprocessing
    # -------------------------------------------------------------
    story.append(Paragraph("4. Dataset, Cleaning & Feature Engineering", h1_style))
    story.append(Paragraph(
        "<b>Dataset Source & Scale:</b><br/>"
        "Humne <b>US Bureau of Transportation Statistics (BTS)</b> ka 3-Million Flight records wala official dataset use kiya. "
        "Isme pure America ke sabhi domestic commercial carriers ka flight performance data shamil hai.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Data Cleaning Steps:</b><br/>"
        "1. <b>Cancelled & Diverted Flights Filter:</b> Jo flights cancel ho gayi ya divert ho gayi, unka actual arrival delay calculate nahi kiya ja sakta, "
        "isliye unhe clean kiya gaya (3,000,000 &rarr; <b>2,913,802 Valid Flights</b>).<br/>"
        "2. <b>Missing Value Audit:</b> Pre-departure schedule columns me missing values virtually 0.00% the (`CRS_ELAPSED_TIME` me 3M me se sirf 14 rows null the, jinpar mean imputation lagaya).<br/>"
        "3. <b>Zero Placeholders Audit:</b> Check kiya gaya ki Distance ya Flight Duration me koi invalid '0' toh nahi hai.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Feature Selection (21,513 Dimensions &rarr; 12 Core Features — 96.4% Reduction):</b><br/>"
        "Pehle high-dimensional one-hot route encoding se 21,513 dimensions ban rahi thi, jisse model heavy ho raha tha aur server par load badh raha tha. "
        "Humne <b>ANOVA F-Test</b> aur feature importance se top 12 features lock kiye jinhone 99.8% predictive accuracy retain ki aur inference latency ko sub-100ms bana diya:",
        body_style
    ))

    anova_data = [
        ['Rank', 'Feature Name', 'ANOVA F-Statistic', 'Operational Significance'],
        ['1', 'CRS_DEP_TIME (Sched Departure)', '1132.51', 'Sabse strong driver: Late evening flights me airport air traffic congestion accumulate hota hai.'],
        ['2', 'CRS_ARR_TIME (Sched Arrival)', '856.61', 'Destination hub airport ke arrival slot banks ke sath directly align karta hai.'],
        ['3', 'CRS_ELAPSED_TIME (Flight Duration)', '29.03', 'Scheduled flight buffer capacity: lambi flights raste me thoda time absorb kar leti hain.'],
        ['4', 'DISTANCE (Flight Distance)', '17.40', 'Short haul feeder routes par turnaround tight hota hai, jabki long haul me throttle recovery buffer hota hai.']
    ]
    t_anova = Table(anova_data, colWidths=[35, 150, 95, 290])
    t_anova.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (2,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_anova)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 5: Model Benchmarking
    # -------------------------------------------------------------
    story.append(Paragraph("5. Systematic Model Benchmarking: 5 Classifiers Compared", h1_style))
    story.append(Paragraph(
        "Humne Machine Learning ke sabhi pramukh paradigms (Linear, Single-Tree, Bagging, Gradient Boosting) ko <b>582,761 untouched test flights</b> par train aur evaluate kiya:",
        body_style
    ))

    model_bench_data = [
        ['#', 'Model Name', 'ML Paradigm', 'Accuracy (%)', 'ROC-AUC (%)', 'PR-AUC (%)', 'Why Chosen / Evaluation Verdict'],
        ['1', 'Logistic Regression', 'Linear Baseline', '91.74%', '63.90%', '13.72%', 'Fastest to train; linear boundary fails on temporal interactions.'],
        ['2', 'Decision Tree', 'Single Tree Rule', '91.46%', '63.53%', '12.96%', 'Interpretable rules but high variance and boundary overfitting.'],
        ['3', 'Random Forest (100 Trees)', 'Bagging Ensemble', '91.77%', '65.51%', '14.36%', 'Robust variance reduction across diverse flight corridors.'],
        ['4', 'LightGBM Classifier', 'Gradient Boosting', '91.78%', '69.24%', '17.52%', 'Histogram-based leaf-wise splitting; extremely fast training.'],
        ['5', 'Tuned XGBoost (Winner)', 'Extreme Grad Boosting', '91.78%', '69.84%', '17.43%', 'Top individual model! Highest ROC-AUC & best probability calibration.']
    ]
    t_models = Table(model_bench_data, colWidths=[20, 130, 95, 65, 65, 60, 135])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#DCFCE7')), # Highlight XGBoost
        ('FONTNAME', (0,5), (-1,5), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,5), (-1,5), colors.HexColor('#166534')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (5,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>XGBoost Kyu Jeeta? (Why XGBoost Won):</b><br/>"
        "XGBoost (Extreme Gradient Boosting) me regularized objective loss function (`L1/L2 regularization`) hota hai "
        "jo tabular categorical data par overfit hone se rokta hai. Isne decision threshold tuning ke dauran sabse smooth probability distribution generate ki, "
        "isliye humne isse production deployment ke liye core engine chuna.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 6: Imbalance & Threshold Tuning
    # -------------------------------------------------------------
    story.append(Paragraph("6. Class Imbalance & The Threshold Tuning Magic (T = 0.12 - 0.15)", h1_style))
    story.append(Paragraph(
        "Ye concept kisi bhi Machine Learning interview me 100 me se 100 marks dilwayega! Isse dhyan se samajhiye:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Default Threshold (T = 0.50) Ka Dhokha (The Trap):</b><br/>"
        "Standard machine learning libraries (jaise Scikit-Learn ya XGBoost) me agar probability &ge; 50% (0.50) hoti hai toh model bolta hai 'Class 1' "
        "aur agar < 50% hoti hai toh bolta hai 'Class 0'.<br/>"
        "Lekin jab dataset me severe delays sirf <b>8.22%</b> hain, toh model ka natural output probability 5% se 30% ke bich ghoomta hai. "
        "Agar aap 50% ka danda laga denge, toh koi flight 50% cross hi nahi karegi! Model sabko On-Time bol dega. "
        "Accuracy toh 91.78% dikhegi, lekin asal late flight pakadne ka <b>Recall sirf 0.09% (Zero)</b> reh jayega!",
        body_style
    ))
    story.append(Paragraph(
        "<b>Threshold Calibration Solution (T = 0.12 ya 12.00%):</b><br/>"
        "Humne validation set par precision-recall curve analyze karke operational threshold ko <b>T = 0.12 (12.00%)</b> par lock kiya:<br/>"
        "• Asal base rate 8.22% hai. Agar model kisi flight ko <b>16.07%</b> delay probability deta hai, iska matlab us flight ka delay risk normal se <b>double (2x)</b> hai!<br/>"
        "• Threshold ko 12% karne se minority delay detection ka <b>Recall 16.4 guna (16.4x) badh gaya</b>, aur model ne 27,889 real severe delays ko successfully pakad liya!",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 7: Hybrid Ensembles & Meta-Stacking
    # -------------------------------------------------------------
    story.append(Paragraph("7. Advanced Hybrid Ensembles & Stacking (92.35% Peak Accuracy)", h1_style))
    story.append(Paragraph(
        "Single models ke baad humne advanced ensemble techniques explore kiye taaki accuracy aur ROC-AUC dono ko push kiya ja sake:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Soft Voting Weighted Blend:</b> XGBoost (45%) + LightGBM (40%) + Random Forest (15%) ki probabilities ka weighted average liya gaya.<br/>"
        "• <b>Level-1 Meta-Stacking:</b> Base models ke 5 Out-Of-Fold (OOF) predicted probability vectors par ek Logistic Regression meta-learner train kiya gaya.<br/>"
        "• <b>HV-4 Unanimous Consensus Model:</b> Jab 5 base learners me se consensus banta hai, toh model achieve karta hai:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Accuracy: 92.35%</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>ROC-AUC: 70.42%</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Precision: 48.15% &nbsp;|&nbsp; Recall: 26.20% &nbsp;|&nbsp; F1-Score: 33.94%</b>",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 8: Explainable AI (SHAP)
    # -------------------------------------------------------------
    story.append(Paragraph("8. Explainable AI (XAI) via SHAP: Model Ke Andar Kya Chal Raha Hai?", h1_style))
    story.append(Paragraph(
        "<b>Model Black-Box Kyu Nahi Hai?</b><br/>"
        "Humne <b>SHAP (SHapley Additive exPlanations)</b> use kiya jo game theory par based hota hai. "
        "Ye mathematically batata hai ki model ne kisi flight ko 'Delayed' ya 'On-Time' kyun bola aur kis feature ne kitna vote diya.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3 Key XAI Operational Insights:</b><br/>"
        "1. <b>Departure Hour Impact (<code>CRS_DEP_HOUR</code>):</b> Mean |SHAP| value > 0.20 ke sath sabse bada predictor hai. "
        "Raat 7:00 PM se 11:00 PM ki flights me SHAP values strongly positive (delay risk badhane wali) hoti hain, kyunki din bhar ke turnaround delays raat me jama ho jaate hain. "
        "Subah 6:00 AM se 9:00 AM ki flights me SHAP values negative hoti hain (protection against delay).<br/>"
        "2. <b>Seasonal Surges (<code>MONTH</code>):</b> Summer holiday season (June/July) aur Thanksgiving/Christmas travel me positive attribution spikes aate hain.<br/>"
        "3. <b>Carrier & Hub Dynamics (<code>AIRLINE</code> & <code>ORIGIN</code>):</b> Crowded hub airports (jaise Newark EWR, Chicago O'Hare ORD, New York JFK) "
        "aur tight turnaround low-cost airlines me baseline delay propensity high hoti hai.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 9: Full-Stack Cloud Deployment
    # -------------------------------------------------------------
    story.append(Paragraph("9. Full-Stack Production Cloud Architecture", h1_style))
    story.append(Paragraph(
        "Zyadatar student ML projects sirf Jupyter Notebook (`.ipynb`) me khatam ho jaate hain. "
        "Humare project ki sabse badi USP ye hai ki humne isse <b>real-world, 24/7 accessible live cloud product</b> me convert kiya hai!",
        body_style
    ))

    arch_box_data = [
        ['Component', 'Technology Used', 'Architecture Details & Real-World Role'],
        ['Backend Engine', 'Flask (Python 3.10)', 'Decoupled REST microservice; memory me lightweight XGBoost load karke sub-100ms API inference deta hai.'],
        ['Production WSGI', 'Gunicorn (Green Unicorn)', 'Multi-worker production server; concurrent HTTP requests handle karta hai bina crash huye.'],
        ['Containerization', 'Docker (python:3.10-slim)', 'Compact 11MB runtime context; .dockerignore se 600MB data exclude kiya; zero dependency conflicts.'],
        ['CI/CD Automation', 'GitHub Repository', 'codewithkanishka01/flight-delay-predictor se connected; git push karte hi automated cloud build trigger hoti hai.'],
        ['Cloud Platform', 'Render.com (Singapore)', 'Global HTTPS public cloud URL: https://flight-delay-predictor-z2ik.onrender.com/'],
        ['Zero Connection Error', '24/7 Always-Awake Bot', 'Automated health-check pings har 10 min me silent keep-alive bhejti hain taaki server sleep na kare.']
    ]
    t_arch = Table(arch_box_data, colWidths=[110, 120, 240])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 10: Live Web App & Test Cases
    # -------------------------------------------------------------
    story.append(Paragraph("10. Live Web Dashboard & Verified Live Test Cases", h1_style))
    story.append(Paragraph(
        "Aap kisi ko bhi live website kholkar demonstration de sakte hain:",
        body_style
    ))
    story.append(Paragraph(
        "<b>🔴 Live Test Case 1: SEVERE DELAY Alert (Crosses 12.00% Threshold)</b><br/>"
        "• <b>Inputs:</b> Date: <code>2026-07-03</code> (July peak holiday) | Airline: <code>Spirit Air Lines</code> | Origin: <code>FLL</code> | Dest: <code>EWR</code> | Sched Dep: <code>09:00 PM</code> (21:00) | Sched Arr: <code>11:55 PM</code> (23:55) | Distance: <code>1065</code> miles.<br/>"
        "• <b>Live Output:</b> Probability = <b>16.07%</b> | Result Badge: <b><font color='#DC2626'>SEVERE DELAY (At least 45 min late)</font></b> 🚨",
        body_style
    ))
    story.append(Paragraph(
        "<b>🟢 Live Test Case 2: ON TIME Verification (Safe Low-Risk Flight)</b><br/>"
        "• <b>Inputs:</b> Date: <code>2026-05-10</code> | Airline: <code>Delta Air Lines Inc.</code> | Origin: <code>MSP</code> | Dest: <code>SEA</code> | Sched Dep: <code>07:00 AM</code> | Sched Arr: <code>09:15 AM</code> | Distance: <code>1399</code> miles.<br/>"
        "• <b>Live Output:</b> Probability = <b>8.71%</b> | Result Badge: <b><font color='#16A34A'>ON TIME / MINIMAL DELAY (&lt; 45 min)</font></b> 🟢",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # CHAPTER 11: Rapid-Fire Viva & Interview Q&A
    # -------------------------------------------------------------
    story.append(Paragraph("11. Rapid-Fire Viva / Interview Q&A (Professor Cheat Sheet)", h1_style))
    story.append(Paragraph(
        "Sir ya external examiner jo sawal puch sakte hain, unka exact Hinglish answer neeche diya gaya hai. "
        "Aap in answers ko natural language me confidently bol sakte hain:",
        body_style
    ))

    qa_list = [
        (
            "Q1: Tumhare project ka exact problem statement kya hai?",
            "Sir, humara objective hai pre-departure flight schedule features dekhkar predict karna ki kya flight 45 minutes ya usse zyada late (Severe Delay) hogi ya on-time pahuchegi. Hum strictly zero post-departure data leakage follow karte hain aur humne isse production cloud web app par deploy kiya hai."
        ),
        (
            "Q2: Standard 15 minute delay chhodkar 45 minute ka cutoff kyu chuna?",
            "Sir, do main reasons hain: Pehla operational reason — airline operations me 45-minute delay critical threshold hota hai jahan connecting flights miss hoti hain, crew duty hours expire hote hain aur FAA ground delay program trigger hota hai. Dusra mathematical reason — 15 minute delay par pre-departure features ka natural ceiling 81.8% hai. 45-minute cutoff par 91.78% flights on-time aur 8.22% severe delay hoti hain, jahan model legitimately 91.78% accuracy achieve karta hai bina majority collapse ke."
        ),
        (
            "Q3: Data Leakage kya hota hai aur tumne isse kaise roka?",
            "Sir, data leakage tab hota hai jab training data me aisi information chali jaye jo real-world prediction ke waqt available na ho. Aviation me departure delay (DEP_DELAY) ka arrival delay se 0.93 correlation hota hai, jisko use karke fake 98% accuracy nikalna aasan hai. Humne DEP_DELAY, TAXI_OUT, ARR_TIME samet sabhi 17 post-departure variables ko permanent drop kiya taaki model 100% genuine pre-boarding data par sikhe."
        ),
        (
            "Q4: Dataset kitna bada tha aur cleaning kaise ki?",
            "Sir, humne US Bureau of Transportation Statistics (BTS) ka official 3-Million flight records dataset use kiya. Cleaning me humne cancelled aur diverted flights filter kiye (kyunki unka arrival delay undefined hota hai), jisse 2,913,802 clean completed flights bachi. Missing values check kiye jo <0.001% the, aur 80/20 stratified train-test split kiya."
        ),
        (
            "Q5: 21,513 features se 12 features par kyu aaye?",
            "Sir, agar categorical flight routes ko blind one-hot encoding karein toh 21,513 sparse dimensions banti hain jo overfitting aur high latency create karti hain. Humne ANOVA F-Test se top 12 pre-departure predictors chune (CRS_DEP_TIME, CRS_ARR_TIME, DISTANCE, AIRLINE, etc.). Is 96.4% dimensionality reduction se model ne 99.8% predictive power retain ki aur inference time घटकर sub-100ms ho gaya."
        ),
        (
            "Q6: Sabhi 5 models ki accuracy ~91.7% ke aas-paas kyu aa rahi thi?",
            "Sir, isse Machine Learning me 'Accuracy Paradox under Class Imbalance' kehte hain. Dataset me exactly 91.78% flights on-time (<45 min) hain. Agar koi dummy model sabhi flights ko Class 0 bol de toh uski accuracy bhi 91.78% aayegi. Lekin unka discrimination power ROC-AUC me dikhta hai: Tuned XGBoost ka ROC-AUC 69.84% hai, jo Logistic Regression se +5.94% better hai."
        ),
        (
            "Q7: Threshold T = 0.50 se T = 0.12 kyu change kiya?",
            "Sir, kyunki severe delay ka base rate sirf 8.22% hai. Default 0.50 threshold par model sabhi flights ko Class 0 predict karta hai aur Recall zero ho jata hai. Humne threshold ko 0.12 par calibrate kiya. Iska matlab agar kisi flight ka predicted risk 12% ya usse zyada hai, toh wo national average (8.22%) se 1.5x zyada risky hai. Is threshold calibration ne minority delay recall ko 16.4x boost kar diya!"
        ),
        (
            "Q8: XGBoost aur Ensembles me kya difference tha?",
            "Sir, single model me Tuned XGBoost best raha (91.78% accuracy, 69.84% ROC). Lekin jab humne XGBoost, LightGBM aur Random Forest ko Level-1 Logistic Regression meta-learner se Stack kiya (Consensus Model), toh usne peak 92.35% accuracy aur 70.42% ROC-AUC achieve kiya."
        ),
        (
            "Q9: Explainable AI (SHAP) se kya pata chala?",
            "Sir, SHAP analysis se pata chala ki scheduled departure hour (CRS_DEP_HOUR) sabse bada feature hai (|SHAP| > 0.20). Shaam 7 PM se 11 PM ki flights me delay risk sabse high hota hai kyunki din bhar ke cascading delays accumulate hote hain, jabki subah 6 AM se 9 AM ki flights me delay risk minimum hota hai."
        ),
        (
            "Q10: Deployment architecture kaisa hai?",
            "Sir, humne Flask Python backend banaya jisme tuned XGBoost model load hota hai. Production server ke liye Gunicorn WSGI use kiya hai, Docker container (python:3.10-slim) banaya hai, GitHub CI/CD se Render.com par deploy kiya hai, aur 24/7 background keep-alive ping set kiya hai taaki cold-start ya connection error na aaye."
        ),
        (
            "Q11: Live website par prediction kaise kaam karti hai?",
            "Sir, user website par jaakar Flight Date, Airline, Origin, Destination, Scheduled Times aur Distance dalta hai. Sub-100ms me API backend feature transformation karke XGBoost model se probability predict karta hai. Agar prob >= 12% ho toh Red Alert 'SEVERE DELAY' show hota hai, aur agar <12% ho toh Green 'ON TIME' badge show hota hai."
        ),
        (
            "Q12: Is project ka future scope kya hai?",
            "Sir, future me hum 3 major cheezein add kar sakte hain: 1) Real-time NOAA METAR weather radar API integration, 2) Aircraft tail-number tracking taaki inbound flight ki physical location se turnaround delay accurately predict ho sake, aur 3) Graph Neural Networks (GNN) se national airport network delay propagation model karna."
        )
    ]

    for q, a in qa_list:
        story.append(Paragraph(q, qa_q_style))
        story.append(Paragraph(a, qa_a_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully compiled at: {pdf_filename_ws}")

    # Copy to Desktop
    pdf_filename_dt = "/Users/kanishkapandey/Desktop/Flight_Delay_Prediction_Complete_Study_Guide_Hinglish.pdf"
    try:
        import shutil
        shutil.copyfile(pdf_filename_ws, pdf_filename_dt)
        print(f"Copied to Desktop at: {pdf_filename_dt}")
    except Exception as e:
        print(f"Desktop copy notice: {e}")

if __name__ == '__main__':
    build_pdf()
