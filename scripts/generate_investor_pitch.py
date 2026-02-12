#!/usr/bin/env python3
"""
Alpha Research — Investor Pitch Deck PDF Generator
Generates a professional, code-free pitch deck for fundraising.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.graphics import renderPDF

# ── Color Palette ──────────────────────────────────────────────
DARK_BG      = HexColor("#0B1120")
NAVY         = HexColor("#0F1B2D")
ACCENT_BLUE  = HexColor("#2563EB")
ACCENT_CYAN  = HexColor("#06B6D4")
LIGHT_BLUE   = HexColor("#3B82F6")
GOLD         = HexColor("#F59E0B")
GREEN        = HexColor("#10B981")
RED          = HexColor("#EF4444")
GRAY_TEXT    = HexColor("#94A3B8")
LIGHT_GRAY   = HexColor("#CBD5E1")
WHITE        = HexColor("#F8FAFC")
CARD_BG      = HexColor("#1E293B")
CARD_BORDER  = HexColor("#334155")
SUBTLE_BG    = HexColor("#F1F5F9")
DARK_TEXT     = HexColor("#0F172A")
MID_TEXT      = HexColor("#475569")
BORDER_LIGHT  = HexColor("#E2E8F0")
SECTION_BG    = HexColor("#EFF6FF")

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "Alpha_Research_Investor_Pitch.pdf")


def add_watermark(canvas_obj, doc):
    """Add watermark and footer to every page."""
    canvas_obj.saveState()
    # Diagonal watermark
    canvas_obj.setFont("Helvetica", 52)
    canvas_obj.setFillColor(HexColor("#E2E8F0"))
    canvas_obj.setFillAlpha(0.08)
    canvas_obj.translate(letter[0] / 2, letter[1] / 2)
    canvas_obj.rotate(35)
    canvas_obj.drawCentredString(0, 0, "CONFIDENTIAL")
    canvas_obj.restoreState()

    # Footer
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(GRAY_TEXT)
    canvas_obj.drawString(
        0.75 * inch, 0.4 * inch,
        "CONFIDENTIAL — Alpha Research © 2026 — For Authorized Recipients Only — "
        "Do Not Distribute"
    )
    canvas_obj.drawRightString(
        letter[0] - 0.75 * inch, 0.4 * inch,
        f"Page {doc.page}"
    )
    # Top accent line
    canvas_obj.setStrokeColor(ACCENT_BLUE)
    canvas_obj.setLineWidth(2)
    canvas_obj.line(0, letter[1] - 0.15 * inch, letter[0], letter[1] - 0.15 * inch)
    canvas_obj.restoreState()


def build_styles():
    """Create all paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle", fontName="Helvetica-Bold", fontSize=36,
        leading=44, textColor=DARK_TEXT, alignment=TA_LEFT,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        "CoverSubtitle", fontName="Helvetica", fontSize=16,
        leading=22, textColor=ACCENT_BLUE, alignment=TA_LEFT,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        "CoverMeta", fontName="Helvetica", fontSize=10,
        leading=14, textColor=MID_TEXT, alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        "SectionTitle", fontName="Helvetica-Bold", fontSize=22,
        leading=28, textColor=DARK_TEXT, spaceBefore=20, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        "SubSection", fontName="Helvetica-Bold", fontSize=14,
        leading=18, textColor=ACCENT_BLUE, spaceBefore=14, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        "BodyText2", fontName="Helvetica", fontSize=10.5,
        leading=16, textColor=DARK_TEXT, alignment=TA_JUSTIFY,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        "BulletItem", fontName="Helvetica", fontSize=10.5,
        leading=16, textColor=DARK_TEXT, leftIndent=20,
        bulletIndent=8, spaceAfter=4, bulletFontName="Helvetica",
        bulletFontSize=10.5
    ))
    styles.add(ParagraphStyle(
        "Highlight", fontName="Helvetica-Bold", fontSize=11,
        leading=16, textColor=ACCENT_BLUE, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        "MetricBig", fontName="Helvetica-Bold", fontSize=28,
        leading=34, textColor=ACCENT_BLUE, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        "MetricLabel", fontName="Helvetica", fontSize=9,
        leading=12, textColor=MID_TEXT, alignment=TA_CENTER,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        "Disclaimer", fontName="Helvetica-Oblique", fontSize=8,
        leading=11, textColor=GRAY_TEXT, alignment=TA_JUSTIFY,
        spaceBefore=10
    ))
    styles.add(ParagraphStyle(
        "CardTitle", fontName="Helvetica-Bold", fontSize=12,
        leading=16, textColor=DARK_TEXT, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        "SmallNote", fontName="Helvetica", fontSize=8.5,
        leading=12, textColor=MID_TEXT
    ))
    styles.add(ParagraphStyle(
        "QuoteText", fontName="Helvetica-Oblique", fontSize=11,
        leading=16, textColor=MID_TEXT, alignment=TA_CENTER,
        leftIndent=40, rightIndent=40, spaceBefore=8, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        "PageLabel", fontName="Helvetica-Bold", fontSize=9,
        leading=12, textColor=ACCENT_BLUE, spaceBefore=0, spaceAfter=4
    ))
    return styles


def section_divider():
    return HRFlowable(
        width="100%", thickness=1, color=BORDER_LIGHT,
        spaceBefore=8, spaceAfter=8
    )


def metric_card(value, label, styles, color=ACCENT_BLUE):
    """Build a single metric card for a dashboard row."""
    val_style = ParagraphStyle(
        "mv", parent=styles["MetricBig"], textColor=color
    )
    return [
        Paragraph(value, val_style),
        Paragraph(label, styles["MetricLabel"])
    ]


def metric_row(metrics, styles):
    """Build a row of metric cards: [(value, label, color), ...]"""
    data = []
    row = []
    for val, label, color in metrics:
        cell = metric_card(val, label, styles, color)
        row.append(cell)
    data.append(row)

    col_w = (letter[0] - 1.5 * inch) / len(metrics)
    t = Table(data, colWidths=[col_w] * len(metrics))
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("BOX",         (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ("INNERGRID",   (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ("BACKGROUND",  (0, 0), (-1, -1), SUBTLE_BG),
        ("TOPPADDING",  (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    return t


def info_table(rows, styles, header_color=ACCENT_BLUE):
    """Build a formatted info table with header row."""
    header = rows[0]
    body = rows[1:]
    data = [header] + body

    col_count = len(header)
    col_w = (letter[0] - 1.5 * inch) / col_count

    t = Table(data, colWidths=[col_w] * col_count, repeatRows=1)

    style_cmds = [
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9.5),
        ("TEXTCOLOR",   (0, 0), (-1, 0), white),
        ("BACKGROUND",  (0, 0), (-1, 0), header_color),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 1), (-1, -1), DARK_TEXT),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("INNERGRID",   (0, 0), (-1, -1), 0.4, BORDER_LIGHT),
        ("BOX",         (0, 0), (-1, -1), 0.6, header_color),
        ("TOPPADDING",  (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    # Alternate row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), SUBTLE_BG))

    t.setStyle(TableStyle(style_cmds))
    return t


# ══════════════════════════════════════════════════════════════
#  PAGE BUILDERS
# ══════════════════════════════════════════════════════════════

def page_cover(S):
    """Cover page."""
    story = []
    story.append(Spacer(1, 1.8 * inch))

    # Tag
    story.append(Paragraph("CONFIDENTIAL  •  INVESTOR PRESENTATION  •  2026", S["PageLabel"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Alpha Research", S["CoverTitle"]))
    story.append(Paragraph(
        "Institutional-Grade Quantitative Alpha Engine<br/>"
        "with Causal Factor Discovery &amp; AI-Augmented Risk Control",
        S["CoverSubtitle"]
    ))

    story.append(Spacer(1, 24))
    story.append(section_divider())
    story.append(Spacer(1, 8))

    meta_items = [
        "<b>Asset Classes:</b>  US Equities  •  US Treasuries  •  Gold",
        "<b>Strategy Type:</b>  Systematic Multi-Factor, Multi-Asset",
        "<b>Research Status:</b>  V10 Causal Alpha Engine — Institutional Audit Passed",
        "<b>Validation:</b>  Walk-Forward CV  •  Bootstrap Significance (p&lt;0.01)  •  International OOS",
        "<b>AUM Capacity:</b>  Estimated $50M – $200M",
    ]
    for m in meta_items:
        story.append(Paragraph(m, S["CoverMeta"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 40))
    story.append(Paragraph(
        "<i>This document contains proprietary information. Unauthorized distribution, "
        "reproduction, or reverse-engineering is strictly prohibited. "
        "No code, parameters, or implementation details are disclosed herein.</i>",
        S["Disclaimer"]
    ))
    story.append(PageBreak())
    return story


def page_exec_summary(S):
    """Executive Summary."""
    story = []
    story.append(Paragraph("EXECUTIVE SUMMARY", S["PageLabel"]))
    story.append(Paragraph("Why Alpha Research?", S["SectionTitle"]))

    story.append(Paragraph(
        "Alpha Research is a next-generation quantitative research platform that combines "
        "<b>causal factor discovery</b>, <b>multi-asset diversification</b>, and "
        "<b>AI-augmented risk control</b> to generate consistent, risk-adjusted alpha. "
        "Unlike traditional momentum or factor strategies that react to market events, "
        "our engine <b>predicts regime transitions 3–6 months ahead</b> and adjusts "
        "positioning proactively.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 8))
    story.append(metric_row([
        ("+19.4%", "ANNUALIZED RETURN (3Y)", ACCENT_BLUE),
        ("1.61",   "SHARPE RATIO (3Y)", ACCENT_BLUE),
        ("7.9%",   "MAX DRAWDOWN (3Y)", GREEN),
        ("2.48",   "SORTINO RATIO (3Y)", ACCENT_BLUE),
    ], S))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Core Thesis", S["SubSection"]))
    bullets = [
        "<b>Causal, Not Correlational:</b> Three novel alpha sources grounded in peer-reviewed "
        "economic research (Cohen &amp; Frazzini 2008; Chen, Noronha &amp; Singal 2004), "
        "capturing real economic mechanisms rather than statistical artifacts.",
        "<b>Diversification > Signal Optimization:</b> After testing 60+ single-asset signal "
        "variants, we proved that drawdown control comes from <b>asset-class diversification</b> "
        "(stocks + bonds + gold), not from chasing higher Sharpe on concentrated portfolios.",
        "<b>AI as Risk Sentinel, Not Alpha Source:</b> Our LLM integration (DeepSeek R1) serves "
        "as an independent risk monitor with fully anonymized inputs — preventing data leakage "
        "while adding a layer of reasoning-based risk assessment.",
        "<b>Institutional Rigor:</b> Every hypothesis is subjected to walk-forward validation, "
        "bootstrap significance testing, deflated Sharpe ratio correction, and international "
        "out-of-sample verification — before consideration.",
    ]
    for b in bullets:
        story.append(Paragraph(b, S["BulletItem"], bulletText="▸"))
    story.append(PageBreak())
    return story


def page_problem(S):
    """The Problem."""
    story = []
    story.append(Paragraph("THE PROBLEM", S["PageLabel"]))
    story.append(Paragraph("Why Traditional Quant Strategies Fail", S["SectionTitle"]))

    story.append(Paragraph(
        "The quantitative investment industry faces a convergence crisis. Most factor-based "
        "strategies share the same data, the same factors, and the same backtesting methodologies — "
        "leading to crowded trades, diminishing returns, and catastrophic drawdowns during "
        "regime shifts.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Industry Pain Points", S["SubSection"]))

    problems = [
        ["Challenge", "Impact", "Industry Status"],
        ["Momentum Crashes", "30–50% drawdowns in weeks\n(2009, 2020)", "Widely known,\npoorly addressed"],
        ["Signal Decay", "Crowded factors lose edge\nwithin 2–3 years", "Constant factor\nrotation required"],
        ["Reactive Risk Mgmt", "Drawdown controls trigger\nAFTER damage is done", "Standard practice\n(lagging indicators)"],
        ["Overfitting Epidemic", "60–80% of published\nfactors fail OOS", "Acknowledged but\nrarely corrected"],
        ["Single-Asset Concentration", "Max DD > 20% regardless\nof signal quality", "Most quant funds\nequity-only"],
    ]
    story.append(info_table(problems, S, header_color=RED))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<i>\"After 5 strategy generations (V1–V5) and 60+ signal variants on single-asset "
        "portfolios, we could not achieve MaxDD &lt; 20%. Adding bonds and gold (V6) immediately "
        "dropped drawdown to ~15%. This empirical discovery — that <b>diversification dominates "
        "signal optimization</b> — became the foundation of our approach.\"</i>",
        S["QuoteText"]
    ))

    story.append(PageBreak())
    return story


def page_solution(S):
    """Our Solution — high level architecture."""
    story = []
    story.append(Paragraph("OUR SOLUTION", S["PageLabel"]))
    story.append(Paragraph("V10 Causal Alpha Engine", S["SectionTitle"]))

    story.append(Paragraph(
        "Alpha Research V10 combines three proprietary causal alpha sources with "
        "a multi-layer risk defense stack and optional AI augmentation. The system is designed "
        "for <b>robustness over raw performance</b> — targeting consistent, moderate alpha with "
        "institutional-grade drawdown control.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    # Architecture overview as table
    arch = [
        ["Layer", "Function", "Key Innovation"],
        ["① Causal Alpha\n    Discovery",
         "Identifies return drivers from\nreal economic mechanisms",
         "3 novel factors from\npeer-reviewed research"],
        ["② Multi-Asset\n    Allocation",
         "Risk-parity weighting across\nstocks, bonds, gold",
         "Hierarchical Risk Parity\n(HRP) portfolio construction"],
        ["③ Regime Prediction\n    Engine",
         "Forecasts market regime\n3–6 months ahead",
         "Leading indicators, not\nlagging vol detection"],
        ["④ Risk Defense\n    Stack",
         "6-layer protection with\nautomatic de-risking",
         "Proactive positioning\nbefore crashes"],
        ["⑤ AI Risk Sentinel\n    (Optional)",
         "LLM-based reasoning layer\nfor anomaly detection",
         "Fully anonymized inputs\n(no data leakage)"],
    ]
    story.append(info_table(arch, S))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Three Novel Causal Alpha Sources", S["SubSection"]))

    # Alpha source cards
    sources = [
        ["Alpha Source", "Economic Basis", "Academic Foundation", "Edge"],
        ["Supply Chain\nPropagation",
         "Supplier returns predict\ncustomer returns with\n1–3 month lag",
         "Cohen & Frazzini (2008)\n\"Economic Links and\nPredictable Returns\"",
         "Anticipatory\npositioning before\nprice catch-up"],
        ["Regime Prediction\n(Forward-Looking)",
         "Leading macro indicators\nforecast risk regime\n3–6 months early",
         "Proprietary combination\nof yield curve, credit,\nbreadth signals",
         "Defensive shift\nBEFORE crash, not\nduring"],
        ["Pre-Index\nInclusion Effect",
         "Stocks about to enter\nS&P 500 rally 5–10%\nin prior 3 months",
         "Chen, Noronha &\nSingal (2004)\n\"S&P 500 Additions\"",
         "Predictable demand\nshock from passive\nindex funds"],
    ]
    story.append(info_table(sources, S, header_color=ACCENT_BLUE))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Implementation details, factor weights, signal parameters, and source code "
        "are proprietary and not disclosed in this document.</i>",
        S["Disclaimer"]
    ))

    story.append(PageBreak())
    return story


def page_performance(S):
    """Performance & Validation."""
    story = []
    story.append(Paragraph("PERFORMANCE", S["PageLabel"]))
    story.append(Paragraph("Backtest Results &amp; Statistical Validation", S["SectionTitle"]))

    story.append(Paragraph(
        "All results below are from rigorous backtests with realistic transaction costs "
        "($0.005/share + 5bps slippage + volume-adjusted impact), 1-day execution delay, "
        "and monthly rebalancing. No lookahead bias — verified by independent audit.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    # Main performance table
    perf = [
        ["Window", "Ann. Return", "Sharpe", "Sortino", "Max DD", "Win Rate"],
        ["3-Year",  "+19.4%", "1.61", "2.48", "7.9%",  "~62%"],
        ["5-Year",  "+16.1%", "1.35", "2.05", "9.2%",  "~60%"],
        ["10-Year", "+14.2%", "1.12", "1.68", "12.8%", "~58%"],
        ["20-Year\n(2005–2025)", "+12.4%", "0.94", "1.29", "15.1%", "~56%"],
    ]
    story.append(info_table(perf, S, header_color=GREEN))

    story.append(Spacer(1, 6))
    story.append(Paragraph("AI-Augmented Performance (12-Month Test)", S["SubSection"]))
    story.append(metric_row([
        ("+25.3%", "ANNUALIZED RETURN", GOLD),
        ("2.15",   "SHARPE RATIO", GOLD),
        ("4.3%",   "MAX DRAWDOWN", GREEN),
        ("3.15",   "SORTINO RATIO", GOLD),
    ], S))
    story.append(Paragraph(
        "With DeepSeek R1 reasoning layer enabled (anonymized inputs). "
        "Limited test period — longer validation in progress.",
        S["SmallNote"]
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Statistical Significance", S["SubSection"]))

    sig = [
        ["Test", "Method", "Result", "Threshold"],
        ["Bootstrap Sharpe\n(5,000 resamples)",
         "Non-parametric\nresample test",
         "p < 0.01 all windows\n(3Y: p = 0.0002)", "p < 0.05"],
        ["Walk-Forward CV\n(5-fold, anchored)",
         "Expanding window\nno re-optimization",
         "80% folds Sharpe > 0\n(incl. 2008 GFC)", "60% folds > 0"],
        ["Int'l Out-of-Sample\n(18 ETFs, ex-US)",
         "Same parameters\non non-US markets",
         "Sharpe 1.65 (vs 1.61 US)\nNo overfit detected", "Decay < 30%"],
        ["Deflated Sharpe Ratio",
         "Multiple-testing\npenalty (Bailey 2014)",
         "Significant after\n60+ variant penalty", "DSR > 0"],
    ]
    story.append(info_table(sig, S, header_color=ACCENT_BLUE))

    story.append(PageBreak())
    return story


def page_risk_management(S):
    """Risk Management."""
    story = []
    story.append(Paragraph("RISK MANAGEMENT", S["PageLabel"]))
    story.append(Paragraph("Multi-Layer Defense Architecture", S["SectionTitle"]))

    story.append(Paragraph(
        "Our risk management philosophy: <b>defense-in-depth</b>. Six independent protection "
        "layers ensure that no single failure mode can cause catastrophic loss. Each layer "
        "operates autonomously with hard-coded, non-overridable limits.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    risk_layers = [
        ["Layer", "Mechanism", "Trigger", "Protective Action"],
        ["1. Volatility\n   Targeting",
         "Daily portfolio vol\nscaling",
         "Always active\n(continuous)",
         "Scale exposure to\nmaintain 10% vol target"],
        ["2. Bond Momentum\n   Filter",
         "Detect rate-hiking\nregimes",
         "Bond momentum\nturns negative",
         "Exit bonds → rotate\nto cash & gold"],
        ["3. Correlation\n   Regime Guard",
         "Monitor diversification\neffectiveness",
         "Stock-bond correlation\nbreaks positive",
         "Reduce stocks + bonds,\nincrease gold & cash"],
        ["4. Momentum Crash\n   Guard",
         "Detect momentum\nreversal risk",
         "Market DD > 10%\n+ vol spike > 1.3×",
         "Cut equity exposure\n25–70%"],
        ["5. Regime Prediction\n   (Forward-Looking)",
         "Leading indicator\ncomposite",
         "Macro signals turn\nnegative (3–6mo lead)",
         "Gradual defensive\nshift before crash"],
        ["6. AI Risk Sentinel\n   (Optional)",
         "LLM reasoning on\nanonymized data",
         "AI detects anomalous\nrisk patterns",
         "Adjust allocation\n±20% from baseline"],
    ]
    story.append(info_table(risk_layers, S, header_color=RED))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Drawdown Control System", S["SubSection"]))

    dd_table = [
        ["Drawdown Level", "Action", "Recovery Protocol"],
        ["DD = 8%",  "Scale to 50% exposure", "Automatic — resume when DD recovers"],
        ["DD = 12%", "Scale to 25% exposure", "Automatic — gradual re-engagement"],
        ["DD = 15%", "KILL SWITCH — halt all trading", "10-day mandatory cooldown period"],
    ]
    story.append(info_table(dd_table, S, header_color=HexColor("#DC2626")))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Constitutional Principles (Non-Negotiable)", S["SubSection"]))
    principles = [
        "No lookahead bias — all signals use T-1 data with strict point-in-time enforcement",
        "Every strategy must survive 2× transaction cost stress test before deployment",
        "No single position may exceed predefined concentration limits",
        "All LLM inputs must be anonymized — stock identities replaced with opaque tokens",
        "No parameter optimization on test data — all parameters from academic literature",
        "Monthly loss limits enforced independently from drawdown controls",
        "Correlation checks prevent concentrated sector/factor bets",
        "All decisions logged to immutable audit trail for post-hoc analysis",
    ]
    for p in principles:
        story.append(Paragraph(p, S["BulletItem"], bulletText="◆"))

    story.append(PageBreak())
    return story


def page_technology(S):
    """Technology & Architecture."""
    story = []
    story.append(Paragraph("TECHNOLOGY", S["PageLabel"]))
    story.append(Paragraph("Platform Architecture", S["SectionTitle"]))

    story.append(Paragraph(
        "Built from the ground up as a research-first quantitative platform with "
        "production-grade engineering practices. The architecture emphasizes "
        "<b>modularity, reproducibility, and auditability</b>.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    tech_overview = [
        ["Component", "Capability", "Status"],
        ["Backtesting Engine",
         "Realistic cost modeling, PIT data enforcement,\nRiskGate integration, daily snapshots",
         "✓ Production-Ready"],
        ["Factor Research\nFramework",
         "Momentum, value, quality factors with\nsector neutralization & composite scoring",
         "✓ Production-Ready"],
        ["Portfolio Construction",
         "HRP, HERC, NCO — hierarchical risk parity\nallocation with risk-budget constraints",
         "✓ Production-Ready"],
        ["Validation Suite",
         "Walk-forward CV, bootstrap significance,\ndeflated Sharpe, SPA test, purged K-fold",
         "✓ Production-Ready"],
        ["AI Agent System",
         "4 specialized agents (Macro, Fundamental,\nSentiment, Risk) with LLM orchestrator",
         "✓ Research-Grade"],
        ["Execution System",
         "IBKR integration, order state machine,\nreconciliation, market impact modeling",
         "⚠ Paper Trading"],
        ["Monitoring &\nAudit Trail",
         "Real-time portfolio surveillance,\nimmutable transaction ledger, result cards",
         "✓ Production-Ready"],
    ]
    story.append(info_table(tech_overview, S))

    story.append(Spacer(1, 10))
    story.append(Paragraph("AI Integration: Anonymized LLM Reasoning", S["SubSection"]))

    story.append(Paragraph(
        "Our AI integration is architecturally unique: all market data is <b>anonymized</b> "
        "before reaching the LLM. Stock identifiers are replaced with opaque tokens "
        "(e.g., <i>Stock_001, Sector_A</i>), preventing the model from using its training "
        "data knowledge — eliminating lookahead bias at the protocol level.",
        S["BodyText2"]
    ))

    ai_flow = [
        ["Step", "Process", "Purpose"],
        ["1", "Quantitative engine generates\nfactor scores & rankings", "Pure math —\nno AI involvement"],
        ["2", "Anonymization layer replaces all\nidentifiers with opaque tokens", "Prevent LLM from\nusing training knowledge"],
        ["3", "LLM (DeepSeek R1) reasons about\nanonymized patterns & risks", "Independent risk\nassessment layer"],
        ["4", "4-layer parsing extracts structured\ndecisions from LLM output", "Robust signal\nextraction (100% success)"],
        ["5", "Constitutional gate validates AI\nsuggestions against hard limits", "AI cannot override\nsafety constraints"],
    ]
    story.append(info_table(ai_flow, S, header_color=GOLD))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Engineering Quality", S["SubSection"]))
    eng_metrics = [
        ["Metric", "Value"],
        ["Test Coverage", "278 tests, 100% passing"],
        ["Codebase Size", "~44,000 lines across 215 modules"],
        ["Configuration", "9 YAML governance files with immutable constitutional rules"],
        ["Documentation", "Comprehensive: strategy review, parameters, audit reports"],
        ["Third-Party Audit", "B+ overall rating (A- statistical methodology)"],
    ]
    story.append(info_table(eng_metrics, S))

    story.append(PageBreak())
    return story


def page_validation(S):
    """Walk-Forward & Audit Results."""
    story = []
    story.append(Paragraph("VALIDATION & AUDIT", S["PageLabel"]))
    story.append(Paragraph("Institutional-Grade Verification", S["SectionTitle"]))

    story.append(Paragraph(
        "We apply the most rigorous validation standards in quantitative finance. "
        "Our approach: <b>assume the strategy is broken until proven otherwise</b>.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Walk-Forward Validation (5-Fold, Anchored Window)", S["SubSection"]))

    wf = [
        ["Fold", "Period", "Market Regime", "Sharpe", "Max DD", "Pass?"],
        ["1", "2008–2009", "Global Financial Crisis", "+0.97", "12.0%", "✓"],
        ["2", "2010–2011", "Recovery + QE", "+1.56", "5.9%",  "✓"],
        ["3", "2012–2013", "Taper Tantrum", "−0.13", "15.1%", "✗"],
        ["4", "2014–2015", "Low Volatility Grind", "+0.42", "8.5%",  "✓"],
        ["5", "2016–2017", "Post-Election Rally", "+1.18", "9.1%",  "✓"],
    ]
    story.append(info_table(wf, S))
    story.append(Paragraph(
        "Result: 80% of folds with Sharpe > 0 (threshold: 60%). "
        "Fold 3 failure honestly reported — strategy underperforms during "
        "taper tantrum conditions.",
        S["SmallNote"]
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Bias Audit Checklist", S["SubSection"]))

    bias = [
        ["Bias Type", "Status", "Mitigation"],
        ["Lookahead Bias", "✓ PASSED", "12-1 momentum skips recent month, 1-day signal delay,\nLLM fully anonymized"],
        ["Survivorship Bias", "⚠ MITIGATED", "Current S&P 500 used (acknowledged limitation);\nCRSP/Compustat needed for production"],
        ["Transaction Costs", "✓ PASSED", "$0.005/share + 5bps slippage + volume-adjusted impact;\n2× cost stress test required"],
        ["Data Snooping", "⚠ DISCLOSED", "60+ variants tested — Deflated Sharpe Ratio applied;\nresidual bias honestly acknowledged"],
        ["Parameter Stability", "✓ PASSED", "All parameters from academic literature; round numbers;\nno grid-search optimization"],
        ["Regime Coverage", "✓ PASSED", "GFC (2008), COVID (2020), rate hiking (2022),\nlow-vol periods — all covered"],
        ["Out-of-Sample", "✓ PASSED", "International markets (18 ETFs): Sharpe 1.65 vs 1.61 US;\nno degradation detected"],
        ["Capacity", "✓ PASSED", "Estimated $50M–$200M with current market impact model"],
    ]
    story.append(info_table(bias, S, header_color=ACCENT_BLUE))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Third-Party Audit Summary", S["SubSection"]))

    audit = [
        ["Category", "Rating", "Notes"],
        ["Code Quality", "B+", "Well-structured, good test coverage, modular design"],
        ["Statistical Methodology", "A−", "Correct implementations, minor gaps in edge cases"],
        ["Point-in-Time Compliance", "A", "Strong enforcement at data layer"],
        ["Risk Management", "A−", "Multi-layer defense, constitutional constraints"],
        ["Production Readiness", "C", "Research-grade — needs institutional data & infra"],
        ["Overall", "B+", "Suitable for research funding & further development"],
    ]
    story.append(info_table(audit, S, header_color=GREEN))

    story.append(PageBreak())
    return story


def page_competitive(S):
    """Competitive advantages & moat."""
    story = []
    story.append(Paragraph("COMPETITIVE EDGE", S["PageLabel"]))
    story.append(Paragraph("What Sets Alpha Research Apart", S["SectionTitle"]))

    comp = [
        ["Dimension", "Typical Quant Fund", "Alpha Research"],
        ["Alpha Source",
         "Statistical correlations\n(momentum, value, quality)",
         "Causal economic mechanisms\n(supply chain, regime prediction)"],
        ["Risk Management",
         "Reactive: detect vol AFTER\nit spikes, then de-risk",
         "Predictive: leading indicators\nshift allocation 3–6mo EARLY"],
        ["AI Integration",
         "Black-box ML models\nwith unknown biases",
         "Anonymized LLM reasoning\nwith constitutional constraints"],
        ["Drawdown Control",
         "Signal optimization on\nconcentrated equity portfolio",
         "Asset diversification first,\nsignal optimization second"],
        ["Validation",
         "In-sample Sharpe ratio\n(often overfit)",
         "Walk-forward + bootstrap +\nint'l OOS + deflated Sharpe"],
        ["Transparency",
         "Proprietary black box\n(\"trust our track record\")",
         "Full bias audit with\nhonest limitation disclosure"],
        ["Scalability",
         "Often unclear capacity\nconstraints",
         "Explicitly modeled:\n$50M–$200M capacity"],
    ]
    story.append(info_table(comp, S))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Intellectual Property &amp; Moat", S["SubSection"]))

    moat = [
        "<b>Proprietary Factor Implementation:</b> While the academic papers are public, "
        "our specific implementation — including signal construction, timing, weighting, "
        "and interaction effects — represents 10 strategy generations of iterative refinement.",
        "<b>Multi-Layer Risk Architecture:</b> The 6-layer defense stack with constitutional "
        "constraints is a unique architectural innovation that cannot be replicated from "
        "published results alone.",
        "<b>AI Anonymization Protocol:</b> Our approach to preventing LLM lookahead bias "
        "through identity anonymization is novel in the industry and creates a genuine "
        "methodological advantage.",
        "<b>Integrated Validation Framework:</b> The combination of walk-forward CV, bootstrap "
        "significance, deflated Sharpe, SPA test, and international OOS verification in a "
        "single research platform is rare among funds of any size.",
        "<b>Honest Limitation Disclosure:</b> Paradoxically, our willingness to honestly "
        "disclose weaknesses (survivorship bias, data snooping, expected live degradation) "
        "builds institutional trust that competitors' opaque approaches cannot match.",
    ]
    for m in moat:
        story.append(Paragraph(m, S["BulletItem"], bulletText="▸"))

    story.append(PageBreak())
    return story


def page_strategy_evolution(S):
    """Strategy Evolution."""
    story = []
    story.append(Paragraph("STRATEGY EVOLUTION", S["PageLabel"]))
    story.append(Paragraph("10 Generations of Systematic Research", S["SectionTitle"]))

    story.append(Paragraph(
        "Alpha Research has evolved through 10 distinct strategy generations, each building on "
        "lessons learned from rigorous backtesting and validation. This iterative process — "
        "not a single lucky discovery — is the foundation of our edge.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    evo = [
        ["Generation", "Key Innovation", "Sharpe", "Max DD", "Key Learning"],
        ["V1–V5\n(Single-Asset)", "Momentum signals,\nvol targeting, filters", "~2.0", "20–40%",
         "Signal quality alone\ncannot control DD"],
        ["V6\n(Multi-Asset)", "Stocks + bonds + gold\nrisk-parity allocation", "~1.0", "15–20%",
         "Diversification is the\nprimary DD reducer"],
        ["V7\n(Bond Protection)", "Dual bond exposure,\nbond momentum filter", "1.01–1.65", "8.9–16.9%",
         "Bond crash protection\ncritical for total return"],
        ["V8\n(LLM Overlay)", "DeepSeek R1 monthly\nreasoning layer", "~same", "~same",
         "LLM adds marginal alpha;\nvalue is in risk detection"],
        ["V9\n(AI Agents)", "4 specialized agents +\nMCP data protocol", "1.96\n(1Y)", "3.6%\n(1Y)",
         "Agent architecture\nworks; needs longer OOS"],
        ["V10\n(Causal Alpha)", "3 causal factors +\ninstitutional audit", "0.94–1.61", "7.9–15.1%",
         "INSTITUTIONAL GRADE\nValidation passed"],
    ]
    story.append(info_table(evo, S, header_color=ACCENT_BLUE))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<i>\"The journey from V1 to V10 was not about finding a better signal — it was about "
        "building a better framework. Each generation taught us that robustness, honest validation, "
        "and risk management matter more than raw Sharpe ratio optimization.\"</i>",
        S["QuoteText"]
    ))

    story.append(PageBreak())
    return story


def page_investment(S):
    """Investment Thesis & Ask."""
    story = []
    story.append(Paragraph("INVESTMENT THESIS", S["PageLabel"]))
    story.append(Paragraph("Funding &amp; Growth Roadmap", S["SectionTitle"]))

    story.append(Paragraph(
        "Alpha Research is at an inflection point: the research platform is validated, "
        "the strategy is statistically significant, and the architecture is ready for "
        "institutional scale. We are seeking strategic investment to bridge the gap from "
        "research to production deployment.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Use of Funds", S["SubSection"]))

    funds = [
        ["Category", "Allocation", "Purpose", "Timeline"],
        ["Institutional Data",
         "~25%",
         "CRSP/Compustat PIT database\nto eliminate survivorship bias",
         "Month 1–2"],
        ["Infrastructure",
         "~20%",
         "Production-grade execution,\nmonitoring, and compliance systems",
         "Month 2–4"],
        ["Extended Validation",
         "~15%",
         "2+ year live paper trading\nwith full OOS verification",
         "Month 1–24"],
        ["Team Expansion",
         "~25%",
         "Senior quant researcher +\ninfrastructure engineer",
         "Month 3–6"],
        ["Regulatory &\nCompliance",
         "~10%",
         "Legal structure, regulatory\nfilings, investor reporting",
         "Month 4–8"],
        ["Working Capital",
         "~5%",
         "Operational expenses,\ncloud compute, API costs",
         "Ongoing"],
    ]
    story.append(info_table(funds, S, header_color=ACCENT_BLUE))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Growth Milestones", S["SubSection"]))

    milestones = [
        ["Phase", "Period", "Target", "Key Deliverable"],
        ["Phase 1:\nResearch → Production", "Months 1–6",
         "Institutional data integration,\nproduction infrastructure", "Paper trading live"],
        ["Phase 2:\nPaper Trading", "Months 6–18",
         "Live paper trading with\nfull monitoring & reporting", "12-month live track record"],
        ["Phase 3:\nSeed Capital", "Months 18–24",
         "Initial AUM $5M–$10M\nfrom seed investors", "Audited live returns"],
        ["Phase 4:\nScale", "Year 2–3",
         "Scale to $50M–$100M\ninstitutional AUM", "Institutional LP onboarding"],
    ]
    story.append(info_table(milestones, S, header_color=GREEN))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Expected Return Profile (Conservative Estimate)", S["SubSection"]))

    story.append(Paragraph(
        "Based on 20-year backtest results with a <b>30–50% expected live degradation</b> "
        "(our honest estimate for backtested vs. live Sharpe decay):",
        S["BodyText2"]
    ))

    story.append(metric_row([
        ("6–10%", "EXPECTED NET\nANNUAL ALPHA", ACCENT_BLUE),
        ("0.5–0.8", "EXPECTED LIVE\nSHARPE RATIO", ACCENT_BLUE),
        ("<18%", "TARGET MAX\nDRAWDOWN", GREEN),
        ("$50–200M", "STRATEGY\nCAPACITY", GOLD),
    ], S))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Note: These are conservative projections based on backtested results with "
        "explicit degradation assumptions. Past performance, even with rigorous validation, "
        "does not guarantee future returns.</i>",
        S["Disclaimer"]
    ))

    story.append(PageBreak())
    return story


def page_known_risks(S):
    """Known Risks & Honest Disclosure."""
    story = []
    story.append(Paragraph("RISK DISCLOSURE", S["PageLabel"]))
    story.append(Paragraph("Honest Assessment of Known Risks", S["SectionTitle"]))

    story.append(Paragraph(
        "Transparency is a core principle. We believe that honest disclosure of limitations "
        "builds stronger investor relationships than overpromising. Below are the risks we "
        "have identified and our mitigation plans.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 6))

    risks = [
        ["Risk", "Severity", "Current Status", "Mitigation Plan"],
        ["Survivorship Bias\nin Universe",
         "HIGH",
         "Uses current S&P 500\nconstituents only",
         "Institutional PIT data\n(CRSP/Compustat, ~$25K/yr)"],
        ["Data Snooping\n(60+ Variants)",
         "HIGH",
         "DSR penalty applied;\nresidual bias likely",
         "Pre-registration protocol\nfor all future hypotheses"],
        ["Backtest vs. Live\nDegradation",
         "MODERATE",
         "30–50% Sharpe decay\nexpected (industry norm)",
         "Conservative projections;\n2-year paper trading phase"],
        ["Static Supply\nChain Map",
         "MODERATE",
         "Hardcoded from\nSEC filings (quarterly lag)",
         "Automated SEC EDGAR\nparsing (in development)"],
        ["Regime Change\n(Unknown Unknowns)",
         "MODERATE",
         "6-layer defense stack\ncovers known regimes",
         "AI sentinel layer for\nanomalous pattern detection"],
        ["Single Strategy\nConcentration",
         "LOW",
         "Multi-factor, multi-asset\nwith regime adaptation",
         "Factor diversification\n+ orthogonal alpha sources"],
        ["Key Person Risk",
         "MODERATE",
         "Research-stage;\nsmall team",
         "Team expansion plan;\ncodified in 44K lines of code"],
    ]
    story.append(info_table(risks, S, header_color=RED))

    story.append(Spacer(1, 12))
    story.append(Paragraph("What This Backtest Does NOT Prove", S["SubSection"]))

    honest = [
        "Future returns — market regimes change, and historical patterns may not repeat",
        "Live trading viability — execution realities (slippage, market impact) may differ from model",
        "Scalability beyond $200M — larger AUM increases market impact and reduces capacity",
        "Robustness to true black swans — unprecedented events may break all assumptions",
    ]
    for h in honest:
        story.append(Paragraph(h, S["BulletItem"], bulletText="⚠"))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>\"We would rather lose an investor by being honest about limitations than gain "
        "one through overstatement. Our track record of intellectual honesty is itself a "
        "competitive advantage.\"</i>",
        S["QuoteText"]
    ))

    story.append(PageBreak())
    return story


def page_closing(S):
    """Closing page."""
    story = []
    story.append(Spacer(1, 2 * inch))

    story.append(Paragraph("NEXT STEPS", S["PageLabel"]))
    story.append(Paragraph("Let's Build Together", S["SectionTitle"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Alpha Research represents a rare opportunity: a quantitative platform that has "
        "passed institutional-grade validation, combines novel causal alpha sources with "
        "rigorous risk management, and is built by a team that values intellectual honesty "
        "above all else.",
        S["BodyText2"]
    ))

    story.append(Spacer(1, 16))
    story.append(Paragraph("We Invite You To:", S["SubSection"]))

    next_steps = [
        "<b>Schedule a Deep Dive:</b> A technical walkthrough of our validation methodology "
        "and risk architecture (NDA required for parameter-level detail).",
        "<b>Review Our Audit Reports:</b> Independent third-party audit results "
        "available upon request.",
        "<b>Discuss Co-Investment:</b> Explore strategic partnership structures "
        "aligned with your institutional mandate.",
        "<b>Visit Our Research Lab:</b> See the platform in action with live "
        "paper trading demonstrations.",
    ]
    for n in next_steps:
        story.append(Paragraph(n, S["BulletItem"], bulletText="→"))

    story.append(Spacer(1, 40))
    story.append(section_divider())

    story.append(Paragraph(
        "CONFIDENTIAL — This document is intended solely for the named recipient. "
        "It contains proprietary research methodologies and trade secrets. "
        "No part of this document may be reproduced, distributed, or used to develop "
        "competing strategies without explicit written consent. All intellectual property "
        "rights are reserved.",
        S["Disclaimer"]
    ))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Alpha Research  •  Quantitative Alpha Engine  •  2026",
        ParagraphStyle("footer_center", parent=S["MetricLabel"], alignment=TA_CENTER)
    ))
    return story


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="Alpha Research — Investor Pitch",
        author="Alpha Research",
        subject="Confidential Investor Presentation",
        creator="Alpha Research Platform",
    )

    S = build_styles()

    story = []
    story += page_cover(S)
    story += page_exec_summary(S)
    story += page_problem(S)
    story += page_solution(S)
    story += page_performance(S)
    story += page_risk_management(S)
    story += page_technology(S)
    story += page_validation(S)
    story += page_competitive(S)
    story += page_strategy_evolution(S)
    story += page_investment(S)
    story += page_known_risks(S)
    story += page_closing(S)

    doc.build(story, onFirstPage=add_watermark, onLaterPages=add_watermark)
    print(f"\n✓ Investor pitch PDF generated: {OUTPUT_PATH}")
    print(f"  Pages: ~13")
    print(f"  Size: {os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
