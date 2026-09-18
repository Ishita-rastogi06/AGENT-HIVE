"""Professional Streamlit dashboard for the AgentHive multi-agent workspace."""

import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.auth_db import create_user, verify_user
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.graph import build_graph

HISTORY_FILE = settings.data_dir / "task_history.json"


# ---------- Shared helpers -------------------------------------------------

def apply_custom_styles() -> None:
    """
    Apply AgentHive strict 4-color visual system:
    - Primary Accent: Sapphire (#72B0AB)
    - Dark Neutral: Spruce (#355E58)
    - Light Neutral / Surface: Arctic (#BCDDDC)
    - Warm Background: Lace (#FFEDD1)
    - Text-on-dark: Pure White (#FFFFFF)
    - Text-on-light: Spruce (#355E58)
    - Dark Accent: Near Black (#0F0F0F)
    """
    st.markdown(
        """
        <style>
        :root {
            --sapphire: #72B0AB;
            --sapphire-dark: #5a928d;
            --spruce: #355E58;
            --arctic: #BCDDDC;
            --lace: #FFEDD1;
            --white: #FFFFFF;
            --near-black: #0F0F0F;
            --radius: 8px;
        }

        /* 1. Page Background: Warm Lace base across all views */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: var(--lace) !important;
            color: var(--spruce) !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Hide Streamlit top header toolbar decorations, but PRESERVE sidebar toggle button */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"], #MainMenu, footer {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99999 !important;
            pointer-events: none !important;
        }

        header[data-testid="stHeader"] * {
            pointer-events: auto !important;
        }

        /* Ensure Sidebar is PERMANENTLY visible and cannot be hidden/collapsed out of view */
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: none !important;
            margin-left: 0 !important;
            background-color: var(--spruce) !important;
            border-right: 1px solid var(--sapphire) !important;
            min-width: 260px !important;
            width: 260px !important;
        }

        /* Position Expand button top-left if collapsed state ever triggers */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[aria-label="Expand sidebar"],
        button[aria-label="Collapse sidebar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 9999999 !important;
            background-color: var(--spruce) !important;
            color: var(--white) !important;
            border-radius: 8px !important;
            border: 1px solid var(--sapphire) !important;
            padding: 4px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        }

        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="collapsedControl"] * {
            color: var(--white) !important;
            fill: var(--white) !important;
        }

        .block-container {
            max-width: 1320px;
            padding: 2rem 2.2rem 3.5rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--white) !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            border-radius: var(--radius) !important;
            padding: 0.45rem 0.75rem !important;
            margin-bottom: 0.2rem !important;
            transition: all 0.15s ease !important;
            color: var(--white) !important;
        }

        /* Active Sidebar Radio Nav Option */
        [data-testid="stSidebar"] .stRadio label[data-checked="true"],
        [data-testid="stSidebar"] .stRadio div[aria-checked="true"] {
            background-color: var(--sapphire) !important;
            color: var(--white) !important;
            font-weight: 700 !important;
        }

        /* Inactive Sidebar Radio Nav Option Hover */
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: var(--arctic) !important;
            color: var(--spruce) !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--sapphire) !important;
            margin: 1rem 0 !important;
        }

        /* 3. Typography Scale: Spruce for all headers and body text */
        h1, h2, h3, h4, h5, h6 {
            color: var(--spruce) !important;
            letter-spacing: -0.025em !important;
            font-weight: 800 !important;
        }

        h1 { font-size: 2.2rem !important; margin-bottom: 0.3rem !important; }
        h2 { font-size: 1.35rem !important; margin-top: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        p, span, label, div, caption {
            color: var(--spruce);
            line-height: 1.55;
        }

        /* 4. Primary & Form Submit Buttons: Teal Spruce fill (#355E58), Bold White text (#FFFFFF) */
        .stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stFormSubmitButton"] > button,
        button[data-testid="stFormSubmitButton"],
        button[kind="primaryFormSubmit"],
        button[data-testid="stBaseButton-primaryFormSubmit"] {
            background-color: var(--spruce) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--sapphire) !important;
            border-radius: var(--radius) !important;
            font-weight: 700 !important;
            padding: 0.55rem 1.1rem !important;
            box-shadow: 0 2px 8px rgba(53, 94, 88, 0.15) !important;
            transition: all 0.15s ease !important;
        }

        .stButton > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] *,
        div[data-testid="stFormSubmitButton"] > button *,
        button[data-testid="stFormSubmitButton"] *,
        button[kind="primaryFormSubmit"] *,
        button[data-testid="stBaseButton-primaryFormSubmit"] * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        .stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        button[data-testid="stFormSubmitButton"]:hover,
        button[kind="primaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background-color: var(--sapphire-dark) !important;
            border-color: var(--sapphire-dark) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        /* 5. Secondary / Outline Buttons: White fill, Spruce border and text */
        .stButton > button, .stDownloadButton > button {
            background-color: var(--white) !important;
            color: var(--spruce) !important;
            border: 1px solid var(--spruce) !important;
            border-radius: var(--radius) !important;
            font-weight: 700 !important;
            padding: 0.55rem 1rem !important;
            transition: all 0.15s ease !important;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: var(--arctic) !important;
            color: var(--spruce) !important;
            border-color: var(--spruce) !important;
        }

        .stButton > button *, .stDownloadButton > button * {
            color: inherit !important;
        }

        /* 6. Layout Containers & Spacing Rhythm */
        [data-testid="stHorizontalBlock"] {
            gap: 1.5rem !important;
            align-items: stretch;
        }

        .result-row-gap { height: 1.5rem; }
        .page-section-gap { height: 2rem; }

        /* Streamlit Bordered Container Wrappers */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: var(--radius) !important;
            border: 1px solid var(--arctic) !important;
            background-color: var(--white) !important;
            box-shadow: 0 2px 8px rgba(53, 94, 88, 0.05) !important;
        }

        /* 7. Form Controls: Inputs, Textareas, Selectboxes */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            border-radius: var(--radius) !important;
            border: 1px solid var(--arctic) !important;
            background-color: var(--white) !important;
            color: var(--spruce) !important;
            font-size: 0.95rem !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--sapphire) !important;
            box-shadow: 0 0 0 3px rgba(114, 176, 171, 0.25) !important;
        }

        /* 8. Dashboard Stat Cards */
        .stat-card {
            min-height: 110px;
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-left: 4px solid var(--sapphire);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
            box-sizing: border-box;
            box-shadow: 0 3px 10px rgba(53, 94, 88, 0.08);
        }

        .stat-label {
            color: var(--spruce) !important;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-value {
            color: var(--sapphire) !important;
            font-size: 1.7rem;
            font-weight: 850;
            letter-spacing: -0.03em;
            margin-top: 0.4rem;
        }

        .stat-subtitle {
            color: var(--spruce) !important;
            font-size: 0.76rem;
            opacity: 0.85;
            margin-top: 0.3rem;
        }

        /* 9. Hero Banner */
        .hero {
            position: relative;
            overflow: hidden;
            color: var(--white);
            margin-top: 0.5rem;
            margin-bottom: 2rem !important;
            padding: 2rem 2.2rem;
            border-radius: var(--radius);
            background-color: var(--spruce);
            border: 1px solid var(--sapphire);
            box-shadow: 0 8px 24px rgba(53, 94, 88, 0.2);
        }

        .hero-kicker {
            color: var(--sapphire);
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .hero-title {
            color: var(--white) !important;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 800;
            max-width: 650px;
        }

        .hero-copy {
            color: var(--arctic);
            margin-top: 0.6rem;
            max-width: 590px;
            line-height: 1.5;
        }

        /* 10. Agent Cards */
        .agent-card {
            position: relative;
            min-height: 140px;
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-left: 4px solid var(--sapphire);
            border-radius: var(--radius);
            padding: 1.1rem;
            box-shadow: 0 3px 10px rgba(53, 94, 88, 0.08);
        }

        .agent-icon {
            width: 36px;
            height: 36px;
            border-radius: var(--radius);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: var(--arctic);
            color: var(--spruce);
            font-size: 1.1rem;
            font-weight: 800;
        }

        .agent-name {
            margin-top: 0.6rem;
            color: var(--spruce);
            font-weight: 800;
            font-size: 1rem;
        }

        .agent-desc {
            color: var(--spruce);
            font-size: 0.84rem;
            opacity: 0.9;
            line-height: 1.4;
            margin-top: 0.25rem;
        }

        .agent-status {
            position: absolute;
            top: 1.05rem;
            right: 1.15rem;
            display: inline-flex;
            gap: 6px;
            align-items: center;
            color: var(--white);
            background-color: var(--sapphire);
            border-radius: 20px;
            padding: 0.2rem 0.55rem;
            font-size: 0.68rem;
            font-weight: 800;
        }

        /* 11. Section Labels */
        .section-label {
            color: var(--spruce) !important;
            font-weight: 800;
            letter-spacing: 0.1em;
            font-size: 0.75rem;
            text-transform: uppercase;
            margin: 2rem 0 0.85rem 0 !important;
        }

        .quick-start-label {
            margin-top: 1.8rem !important;
            margin-bottom: 0.15rem !important;
            font-weight: 850 !important;
            font-size: 1.15rem !important;
            letter-spacing: 0.08em !important;
            color: var(--spruce) !important;
            text-transform: uppercase !important;
            padding-left: 0 !important;
        }

        .quick-start-card-container [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding-top: 0.6rem !important;
            padding-bottom: 0.6rem !important;
            margin-top: 0.15rem !important;
        }

        /* 12. Workspace & Answer Panels */
        .workspace-panel, .answer-box {
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-radius: var(--radius);
            padding: 1.35rem;
            box-shadow: 0 3px 10px rgba(53, 94, 88, 0.08);
        }

        /* Ensure ALL code blocks start scrolled to column 0 on the left and wrap/scroll cleanly */
        [data-testid="stCodeBlock"], [data-testid="stCode"], pre, code, .stCode, div[data-testid="stCodeBlock"] pre {
            overflow-x: auto !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            text-align: left !important;
            direction: ltr !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            box-sizing: border-box !important;
            max-width: 100% !important;
            width: 100% !important;
        }

        .stExpander [data-testid="stCodeBlock"],
        .stExpander [data-testid="stCodeBlock"] pre,
        .stExpander [data-testid="stCodeBlock"] code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-x: auto !important;
            text-align: left !important;
            direction: ltr !important;
            margin-left: 0 !important;
            padding-left: 0.8rem !important;
            width: 100% !important;
        }

        .trace-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--arctic);
        }

        .trace-row:last-child { border: 0; }

        .trace-number {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            flex: 0 0 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: var(--arctic);
            color: var(--spruce);
            font-size: 0.75rem;
            font-weight: 800;
        }

        .trace-text {
            color: var(--spruce);
            font-size: 0.9rem;
            font-weight: 650;
        }

        .eyebrow {
            color: var(--sapphire);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        /* 13. Sidebar Branding & System Status */
        .sidebar-brand { padding: 0.65rem 0.2rem 0.8rem; }
        .sidebar-brand-name { font-size: 1.35rem; font-weight: 850; letter-spacing: -0.04em; color: var(--white); }
        .sidebar .system-chip, [data-testid="stSidebar"] .system-chip { color: var(--white) !important; font-size: 0.78rem; padding: 0.1rem 0; }
        .system-chip { color: #1F3835 !important; font-size: 0.85rem; font-weight: 600; display: inline-block; }
        .system-chip b { color: #1F3835 !important; font-weight: 800; }

        /* User Status Pill in Sidebar (Non-nav style) */
        .user-status-pill {
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
            padding: 5px 14px !important;
            background-color: rgba(114, 176, 171, 0.15) !important;
            border: 1px solid rgba(114, 176, 171, 0.3) !important;
            border-radius: 20px !important;
            font-size: 0.78rem !important;
            color: #BCDDDC !important;
            margin-top: 0.4rem !important;
            margin-bottom: 1.2rem !important;
            width: fit-content !important;
            box-shadow: none !important;
        }

        .user-status-pill .status-dot {
            color: #72B0AB !important;
            font-size: 0.65rem !important;
        }

        .user-status-pill .status-text {
            color: #EAE3D2 !important;
            font-weight: 500 !important;
        }

        .user-status-pill .status-text strong {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* 14. Expanders with Gaps */
        [data-testid="stExpander"] {
            background-color: var(--white) !important;
            border: 1px solid var(--arctic) !important;
            border-radius: var(--radius) !important;
            margin-bottom: 1.25rem !important;
        }

        [data-testid="stExpander"] summary {
            background-color: var(--white) !important;
            color: var(--spruce) !important;
            font-weight: 750 !important;
        }

        /* 15. Code Blocks, Trace Trees, & JSON Block Color Fixes */
        [data-testid="stCodeBlock"], [data-testid="stJson"], pre, code {
            background-color: #FAF5EC !important;
            color: #1F3835 !important;
            border: 1px solid #72B0AB !important;
            border-radius: var(--radius) !important;
        }

        [data-testid="stCodeBlock"] *, [data-testid="stJson"] *, pre *, code * {
            color: #1F3835 !important;
        }

        [data-testid="stJson"] span {
            color: #1F3835 !important;
        }

        .stCodeBlock, div[data-testid="stCodeBlock"] pre, div[data-testid="stJson"] pre {
            background-color: #FAF5EC !important;
            color: #1F3835 !important;
        }

        /* 16. Sidebar Radio Dots: Beige/Lace color (#FFEDD1) with Spruce center */
        [data-testid="stSidebar"] [data-testid="stRadioButton"] *,
        [data-testid="stSidebar"] .stRadio *,
        [data-testid="stSidebar"] [data-baseweb="radio"] * {
            accent-color: #FFEDD1 !important;
            caret-color: #FFEDD1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioButton"] label > div:first-child,
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child,
        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child,
        [data-testid="stSidebar"] input[type="radio"] + div,
        [data-testid="stSidebar"] input[type="radio"] ~ div {
            border-color: #FFEDD1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioButton"] label[data-checked="true"] > div:first-child,
        [data-testid="stSidebar"] .stRadio div[aria-checked="true"] > div:first-child,
        [data-testid="stSidebar"] .stRadio label[data-checked="true"] > div:first-child,
        [data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"],
        [data-testid="stSidebar"] input[type="radio"]:checked + div,
        [data-testid="stSidebar"] input[type="radio"]:checked ~ div {
            border-color: #FFEDD1 !important;
            background-color: #FFEDD1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioButton"] label[data-checked="true"] > div:first-child *,
        [data-testid="stSidebar"] .stRadio div[aria-checked="true"] *,
        [data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] *,
        [data-testid="stSidebar"] input[type="radio"]:checked + div *,
        [data-testid="stSidebar"] input[type="radio"]:checked ~ div * {
            background-color: var(--spruce) !important;
            fill: var(--spruce) !important;
            color: var(--spruce) !important;
            border-color: var(--spruce) !important;
        }

        [data-testid="stSidebar"] svg circle, [data-testid="stSidebar"] svg path {
            fill: #FFEDD1 !important;
        }

        /* 15. Featured Overview & Asymmetric Dashboard Cards */
        .featured-stat-card {
            background-color: var(--spruce);
            border: 1px solid var(--sapphire);
            border-radius: var(--radius);
            padding: 1.3rem 1.5rem;
            color: var(--white);
            box-shadow: 0 4px 16px rgba(53, 94, 88, 0.15);
            height: 184px !important;
            min-height: 184px !important;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .stacked-stat-cards {
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            height: 184px !important;
            gap: 12px !important;
            box-sizing: border-box !important;
        }

        .stat-card-horizontal {
            background-color: var(--white) !important;
            border: 1px solid var(--arctic) !important;
            border-left: 4px solid var(--sapphire) !important;
            border-radius: var(--radius) !important;
            padding: 0.85rem 1.4rem !important;
            box-sizing: border-box !important;
            box-shadow: 0 3px 10px rgba(53, 94, 88, 0.08) !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            height: calc(50% - 6px) !important;
            min-height: 86px !important;
        }

        .stat-card-left {
            display: flex !important;
            align-items: center !important;
        }

        .stat-card-left .stat-label {
            color: var(--spruce) !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin: 0 !important;
        }

        .stat-card-right {
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-end !important;
            text-align: right !important;
        }

        .stat-card-right .stat-value {
            color: var(--sapphire) !important;
            font-size: 1.6rem !important;
            font-weight: 850 !important;
            line-height: 1 !important;
            letter-spacing: -0.03em !important;
            margin: 0 !important;
        }

        .stat-card-right .stat-subtitle {
            color: var(--spruce) !important;
            font-size: 0.76rem !important;
            opacity: 0.85 !important;
            margin-top: 0.25rem !important;
            white-space: nowrap !important;
        }

        .latency-container {
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-radius: var(--radius);
            padding: 1.25rem 1.4rem;
            box-shadow: 0 2px 8px rgba(53, 94, 88, 0.05);
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
        }

        .featured-stat-label {
            color: var(--sapphire);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .featured-stat-value {
            color: var(--white);
            font-size: 2.6rem;
            font-weight: 850;
            line-height: 1;
            margin: 0.3rem 0;
            letter-spacing: -0.03em;
        }

        .featured-stat-sub {
            color: var(--arctic);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .featured-network-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--white);
            margin-top: 8px;
        }

        .latency-container {
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-radius: var(--radius);
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 8px rgba(53, 94, 88, 0.05);
        }

        .latency-header {
            color: var(--spruce);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .latency-item {
            background-color: rgba(188, 221, 220, 0.25);
            border: 1px solid var(--arctic);
            border-radius: 6px;
            padding: 0.6rem 0.75rem;
            text-align: center;
        }

        .latency-agent {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--spruce);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .latency-val {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--sapphire);
            margin-top: 0.15rem;
        }

        /* 16. Auth Page Card & Centered Layout */
        .auth-container {
            max-width: 440px;
            margin: 1.25rem auto 3rem auto;
        }

        .auth-card {
            background-color: var(--white);
            border: 1px solid var(--arctic);
            border-radius: var(--radius);
            padding: 2rem 2.25rem;
            box-shadow: 0 6px 20px rgba(53, 94, 88, 0.08);
            text-align: center;
        }

        .auth-brand {
            margin-bottom: 1.25rem;
            text-align: center;
        }

        .auth-brand-logo {
            font-size: 2.2rem;
            line-height: 1;
            margin-bottom: 0.2rem;
        }

        .auth-brand-title {
            font-size: 1.45rem;
            font-weight: 850;
            color: var(--spruce);
            letter-spacing: -0.02em;
        }

        .auth-brand-subtitle {
            font-size: 0.72rem;
            font-weight: 800;
            color: var(--sapphire);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-top: 0.15rem;
        }

        /* Pill-style Segmented Tabs for Auth Screen */
        [data-baseweb="tab-list"] {
            gap: 6px !important;
            background-color: rgba(188, 221, 220, 0.35) !important;
            padding: 4px !important;
            border-radius: 8px !important;
            border: 1px solid var(--arctic) !important;
            margin-bottom: 1.2rem !important;
        }

        [data-baseweb="tab"] {
            flex: 1 !important;
            height: 38px !important;
            border-radius: 6px !important;
            border: none !important;
            background-color: transparent !important;
            color: var(--spruce) !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            transition: all 0.15s ease !important;
            padding: 0 !important;
            justify-content: center !important;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--sapphire) !important;
            color: var(--white) !important;
            box-shadow: 0 2px 6px rgba(53, 94, 88, 0.15) !important;
        }

        [data-baseweb="tab"][aria-selected="true"] * {
            color: var(--white) !important;
        }

        [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Restyle password visibility toggle icon */
        button[aria-label="Show password"],
        button[aria-label="Hide password"],
        [data-testid="stInputPasswordToggle"] {
            background: transparent !important;
            border: none !important;
            color: var(--spruce) !important;
            opacity: 0.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_history_file() -> Path:
    return settings.data_dir / "task_history.json"


def load_task_history() -> list[dict[str, Any]]:
    history_file = _get_history_file()
    if not history_file.exists():
        return []
    try:
        value = json.loads(history_file.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_task_history(history: list[dict[str, Any]]) -> None:
    settings.ensure_runtime_directories()
    history_file = _get_history_file()
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def route_label(agent: str) -> tuple[str, str, str]:
    if agent == "coder":
        return "Coding Specialist", "⌘", "Code generation"
    elif agent == "data":
        return "Data Specialist", "⬡", "Data & SQL analysis"
    elif agent == "writer":
        return "Writer Specialist", "✎", "Content & report writing"
    return "Research Specialist", "◌", "Research & analysis"


def render_stat_card(label: str, value: str | int, subtitle: str = "") -> None:
    """Render a theme-independent dashboard stat card with readable text."""
    subtitle_html = f'<div class="stat-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_agent_card(icon: str, name: str, description: str) -> None:
    st.markdown(
        f'''<div class="agent-card"><div class="agent-icon">{icon}</div>
        <div class="agent-name">{name}</div><div class="agent-desc">{description}</div>
        <div class="agent-status"><span>●</span> ONLINE</div></div>''',
        unsafe_allow_html=True,
    )


def render_trace(trace: list[str]) -> None:
    if not trace:
        st.caption("No workflow events were returned for this task.")
        return
    lines = []
    for index, step in enumerate(trace, 1):
        lines.append(f'<div class="trace-row"><div class="trace-number">{index}</div><div class="trace-text">{html.escape(str(step))}</div></div>')
    st.markdown("".join(lines), unsafe_allow_html=True)


def render_formatted_output(text: str) -> None:
    """Render specialist output with clean markdown prose and native st.code syntax blocks."""
    if not text or not text.strip():
        st.info("No content returned.")
        return

    import re

    # Match fenced code blocks ```[lang]\n...\n```
    pattern = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    pos = 0
    matches = list(pattern.finditer(text))

    if matches:
        for match in matches:
            # Render prose before code block
            prose = text[pos : match.start()].strip()
            if prose:
                st.markdown(prose)

            lang = match.group(1).strip() or "python"
            code_content = match.group(2).strip()
            if code_content:
                st.code(code_content, language=lang)

            pos = match.end()

        # Render trailing prose after last code block
        trailing_prose = text[pos:].strip()
        if trailing_prose:
            st.markdown(trailing_prose)
    else:
        # Render non-backtick code content as a single cohesive code block
        code_keywords = ("SELECT ", "INSERT INTO ", "CREATE TABLE ", "IMPORT ", "DEF ", "SQLITE3", "PANDAS")
        upper_text = text.upper()
        if any(kw in upper_text for kw in code_keywords):
            lang = "python" if ("IMPORT " in upper_text or "DEF " in upper_text or "SQLITE3" in upper_text) else "sql"
            st.code(text.strip(), language=lang)
        else:
            st.markdown(text)


def show_result(result: dict[str, Any], elapsed_seconds: float | None = None) -> None:
    selected = result.get("selected_agent", "research")
    name, icon, task_type = route_label(selected)
    answer = result.get("final_answer") or result.get("specialist_output") or "No answer was returned."
    trace = result.get("trace", [])
    if not isinstance(trace, list):
        trace = []

    wf_status = (result.get("workflow_status") or "").lower()
    rev_decision = (result.get("review") or {}).get("decision", "").lower()

    if wf_status == "escalated" or rev_decision == "escalate":
        review_status = "Escalated"
        card_label = "Escalated run"
        banner_msg = f"Task ESCALATED — routed to the **{name}** and escalated for human review."
        render_banner = st.warning
    elif wf_status in ("failed", "error"):
        review_status = "Failed"
        card_label = "Failed run"
        banner_msg = f"Task FAILED — routed to the **{name}**."
        render_banner = st.error
    elif wf_status in ("paused_for_approval", "paused") or result.get("__interrupt__"):
        review_status = "Paused"
        card_label = "Paused run"
        banner_msg = f"Task PAUSED — routed to the **{name}** for human approval."
        render_banner = st.info
    else:
        review_status = "Passed"
        card_label = "Completed run"
        banner_msg = f"Task completed — routed to the **{name}** and reviewed by the hive."
        render_banner = st.success

    st.markdown(f'<div class="section-label">{card_label}</div>', unsafe_allow_html=True)
    render_banner(banner_msg)
    a, b, c, d = st.columns(4)
    with a:
        render_stat_card("Route", name)
    with b:
        render_stat_card("Task type", task_type)
    with c:
        render_stat_card("Review", review_status)
    with d:
        render_stat_card("Runtime", f"{elapsed_seconds:.1f}s" if elapsed_seconds is not None else "Complete")

    st.markdown('<div class="result-row-gap" style="height: 16px;"></div>', unsafe_allow_html=True)

    # 1. Final response panel (Full Width)
    with st.container(border=True):
        st.markdown('<div class="eyebrow" style="margin-bottom: 12px;">Final response</div>', unsafe_allow_html=True)
        render_formatted_output(str(answer))

    st.markdown('<div class="result-row-gap" style="height: 16px;"></div>', unsafe_allow_html=True)

    # 2. Workflow activity panel (Full Width Stacked Below)
    with st.container(border=True):
        st.markdown('<div class="eyebrow" style="margin-bottom: 12px;">Workflow activity</div>', unsafe_allow_html=True)
        render_trace(trace)


def _compute_dashboard_performance_metrics() -> dict[str, Any]:
    """Compute performance & latency metrics across task history and logs consistently."""
    history = load_task_history()
    total_tasks = len(history)

    agent_runtimes: dict[str, list[float]] = {"research": [], "coder": [], "data": [], "writer": []}
    task_tool_count = 0

    for item in history:
        agent = item.get("selected_agent", "research")
        rt = float(item.get("runtime_seconds", 0.0))
        if agent in agent_runtimes and rt > 0:
            agent_runtimes[agent].append(rt)

        trace = item.get("trace", [])
        if isinstance(trace, list):
            task_tool_count += sum(1 for step in trace if "tool" in str(step).lower() or "execut" in str(step).lower())

    avg_latencies = {
        agent: (sum(rts) / len(rts) if rts else 0.0)
        for agent, rts in agent_runtimes.items()
    }

    # Count unique escalated tasks strictly among workspace history task runs
    workspace_task_ids = {str(item.get("task_id")) for item in history if item.get("task_id")}
    escalated_workspace_tasks = {
        str(item.get("task_id"))
        for item in history
        if item.get("workflow_status") == "escalated"
        or (isinstance(item.get("review"), dict) and item.get("review", {}).get("decision") == "escalate")
    }

    # Check if any workspace task is pending in approval queue
    if workspace_task_ids:
        try:
            from src.orchestration.approval import get_approval_db_conn
            conn = get_approval_db_conn()
            cursor = conn.execute("SELECT DISTINCT task_id FROM approval_queue WHERE status='pending';")
            for row in cursor.fetchall():
                if row[0] and str(row[0]) in workspace_task_ids:
                    escalated_workspace_tasks.add(str(row[0]))
            conn.close()
        except Exception:
            pass

    total_tasks_count = len(workspace_task_ids)
    escalation_count = len(escalated_workspace_tasks)
    escalation_rate = (escalation_count / total_tasks_count * 100.0) if total_tasks_count > 0 else 0.0
    escalation_rate = min(100.0, max(0.0, escalation_rate))

    return {
        "avg_latencies": avg_latencies,
        "tool_count": task_tool_count,
        "escalation_count": escalation_count,
        "escalation_rate": escalation_rate,
        "total_tasks": total_tasks_count,
    }


def open_workspace() -> None:
    """Callback to navigate to the Workspace page when clicked."""
    st.session_state["page_navigation"] = "Workspace"


def render_dashboard_page() -> None:
    st.markdown(
        '''<div class="hero"><div class="hero-kicker">Multi-agent operations</div>
        <div class="hero-title">Bring every complex task to the hive.</div>
        <div class="hero-copy">AgentHive recalls relevant context, selects the right specialist, reviews its work, and saves useful knowledge for the next request.</div></div>''',
        unsafe_allow_html=True,
    )
    history = load_task_history()
    perf = _compute_dashboard_performance_metrics()

    st.markdown('<div class="section-label">Workspace Overview & Operational Health</div>', unsafe_allow_html=True)

    # Asymmetric layout: Featured main card (Left 60%) + Key Secondary Stats (Right 40%)
    col_feat, col_stats = st.columns([1.6, 1.1], gap="large")

    with col_feat:
        st.markdown(
            f'''<div class="featured-stat-card">
                <div>
                    <div class="featured-stat-label">Tasks Completed</div>
                    <div class="featured-stat-value">{perf["total_tasks"]}</div>
                    <div class="featured-stat-sub">Saved workspace runs in local state</div>
                </div>
                <div class="featured-network-badge">
                    <span style="color: #72B0AB; font-size: 0.9rem;">●</span> <b>Agent Network Status:</b> 4 Specialists Online (Research, Coder, Data, Writer)
                </div>
            </div>''',
            unsafe_allow_html=True,
        )

    with col_stats:
        st.markdown(
            f'''<div class="stacked-stat-cards">
                <div class="stat-card-horizontal">
                    <div class="stat-card-left">
                        <div class="stat-label">Total Tool Calls</div>
                    </div>
                    <div class="stat-card-right">
                        <div class="stat-value">{perf["tool_count"]}</div>
                        <div class="stat-subtitle">Audit log recorded</div>
                    </div>
                </div>
                <div class="stat-card-horizontal">
                    <div class="stat-card-left">
                        <div class="stat-label">Escalation Rate</div>
                    </div>
                    <div class="stat-card-right">
                        <div class="stat-value">{perf["escalation_rate"]:.1f}%</div>
                        <div class="stat-subtitle">{perf["escalation_count"]} of {perf["total_tasks"]} runs escalated</div>
                    </div>
                </div>
            </div>''',
            unsafe_allow_html=True,
        )

    # Unified Specialist Latency Panel with 1.5rem horizontal gap between items
    l_res = perf['avg_latencies']['research']
    l_cod = perf['avg_latencies']['coder']
    l_dat = perf['avg_latencies']['data']
    l_wri = perf['avg_latencies']['writer']

    st.markdown(
        f'''<div class="latency-container">
            <div class="latency-header">⚡ SPECIALIST NETWORK AVERAGE LATENCY</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem;">
                <div class="latency-item">
                    <div class="latency-agent">Research</div>
                    <div class="latency-val">{l_res:.2f}s</div>
                </div>
                <div class="latency-item">
                    <div class="latency-agent">Coder</div>
                    <div class="latency-val">{l_cod:.2f}s</div>
                </div>
                <div class="latency-item">
                    <div class="latency-agent">Data</div>
                    <div class="latency-val">{l_dat:.2f}s</div>
                </div>
                <div class="latency-item">
                    <div class="latency-agent">Writer</div>
                    <div class="latency-val">{l_wri:.2f}s</div>
                </div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Your agent team</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    with cols[0]: render_agent_card("⌁", "Supervisor", "Understands each request and dispatches it to the best specialist.")
    with cols[1]: render_agent_card("◌", "Specialist network", "Research and coding specialists produce focused, high-quality work.")
    with cols[2]: render_agent_card("✓", "Reviewer", "Checks the specialist output before a response is delivered.")

    st.markdown('<div class="section-label quick-start-label">Quick start</div>', unsafe_allow_html=True)
    st.markdown('<div class="quick-start-card-container">', unsafe_allow_html=True)
    with st.container(border=True):
        left, right = st.columns([3.5, 1.5])
        with left:
            st.markdown(
                '<div style="margin:0; padding:0;">'
                '<h3 style="margin: 0 0 3px 0 !important; padding: 0 !important; font-size: 1.1rem; font-weight: 800; color: #355E58; line-height: 1.2;">'
                '➔ Start an orchestrated task in the Workspace</h3>'
                '<p style="margin: 0 !important; padding: 0 !important; font-size: 0.85rem; color: #355E58; opacity: 0.85; line-height: 1.3;">'
                'Ask a research question, analyze data, generate code, or execute complex multi-agent workflows.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.button(
                "Go to Workspace ➔",
                type="primary",
                use_container_width=True,
                on_click=open_workspace,
            )
    st.markdown('</div>', unsafe_allow_html=True)


def render_trace_explorer_page() -> None:
    st.title("Trace Explorer")
    st.caption("Unified execution tree joining LangGraph checkpoints, tool-call logs, and human approval events.")

    user_id = st.session_state.get("user_id") or st.session_state.get("active_user_id", "default")
    st.session_state.active_user_id = user_id
    st.markdown(f'<div class="system-chip" style="margin-bottom: 16px;">Active Identity · <b>{html.escape(user_id)}</b></div>', unsafe_allow_html=True)

    history = load_task_history()
    if not history:
        st.info("No task runs found in local history. Execute tasks in the Workspace to explore execution traces.")
        return

    # Task selector dropdown
    task_map = {
        f"Task {idx+1}: {item.get('task', 'Untitled')[:45]}... ({item.get('timestamp', '')[:16]})": item
        for idx, item in enumerate(reversed(history))
    }
    selected_label = st.selectbox("Select task to inspect", list(task_map.keys()))
    selected_task = task_map[selected_label]

    task_id = selected_task.get("task_id") or "default_task"

    from src.orchestration.trace import build_execution_trace, compare_traces, full_replay_task, partial_replay_task
    trace_data = build_execution_trace(task_id, user_id=user_id)
    nodes = trace_data.get("nodes", [])

    # Overview metrics header
    st.markdown('<div class="result-row-gap"></div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1: render_stat_card("Total Duration", f"{trace_data.get('total_duration_seconds', 0.0):.2f}s")
    with t2: render_stat_card("Checkpoints", trace_data.get("total_steps", 0))
    with t3: render_stat_card("Primary Specialist", trace_data.get("selected_agent", "research"))
    with t4: render_stat_card("Task ID", task_id[:12])

    st.markdown('<div class="section-label">Execution Trace Tree</div>', unsafe_allow_html=True)

    # Collapse by default if total steps >= 20
    auto_collapse = len(nodes) >= 20
    if auto_collapse:
        st.caption("Task contains 20+ steps. Tree nodes are collapsed by default.")

    for node in nodes:
        idx = node.get("step_index", 0)
        step_name = node.get("step_name", "checkpoint")
        dur = node.get("duration_seconds", 0.0)
        diff = node.get("state_diff", {})
        details = node.get("details", {})
        children = node.get("children", [])

        expander_title = f"Step {idx}: {step_name.upper()}  ·  {dur:.2f}s  ·  {len(children)} child event(s)"

        with st.expander(expander_title, expanded=not auto_collapse and idx < 3):
            st.markdown(f"**Step Name**: `{step_name}` | **Duration**: `{dur:.3f}s` | **Node ID**: `{node['node_id']}`")

            # LLM Node details
            if step_name == "supervisor":
                st.markdown("**Supervisor Decomposition Plan**:")
                st.json(details.get("plan") or {})
                if details.get("memory_context"):
                    st.caption(f"Recalled Context: {details['memory_context'][:200]}...")
            elif step_name in ("research", "coder", "data", "writer"):
                st.markdown("**Subtask Instruction**:")
                st.write((details.get("subtask") or {}).get("instruction", "None"))
                st.markdown("**Specialist Response**:")
                st.markdown(details.get("specialist_output", "No response."))
            elif step_name == "reviewer":
                st.markdown("**Review Decision & Feedback**:")
                st.json(details.get("review") or {})

            # Child events (tool calls & approval events)
            if children:
                st.markdown("**Child Events (Tool Calls & Human Approvals)**:")
                for child in children:
                    c_type = child.get("node_type")
                    c_name = child.get("step_name")
                    c_det = child.get("details", {})
                    if c_type == "tool_call":
                        st.info(f"🛠️ **{c_name}** ({child.get('duration_seconds', 0.0):.2f}s)\n\n"
                                f"Arguments: `{json.dumps(c_det.get('arguments') or {})}`\n\n"
                                f"Result: {c_det.get('result_summary', 'None')}")
                    elif c_type == "approval_event":
                        st.warning(f"🛑 **{c_name}** — Decision: `{c_det.get('decision')}`\n\n"
                                   f"Trigger: `{c_det.get('trigger_type')}` | Response: {c_det.get('user_response')}")

            # State diff
            if diff:
                with st.popover(f"View State Diff (Step {idx})"):
                    st.json(diff)

    # ── Replay Control Panel ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Replay System & Step Override</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 🔄 Full Task Replay")
        st.caption("Re-execute this task from scratch as a new task run.")
        if st.button("Run Full Replay", type="primary", use_container_width=True):
            with st.spinner("Replaying full task..."):
                new_tid, res = full_replay_task(task_id, user_id=user_id)
                st.session_state.replay_result = {"new_task_id": new_tid, "result": res, "mode": "full", "orig_task_id": task_id}
                st.success(f"Full replay complete (New Task ID: `{new_tid[:12]}`).")
                st.rerun()

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### ✏️ Partial Replay (Step Override)")
        st.caption("Edit a specific step's output and resume graph execution from that checkpoint.")

        ckpt_options = {f"Step {n['step_index']}: {n['step_name']} ({n['node_id'][:8]})": n for n in nodes if n.get("node_type") == "checkpoint"}
        if ckpt_options:
            sel_ckpt_label = st.selectbox("Select checkpoint step to override", list(ckpt_options.keys()))
            sel_node = ckpt_options[sel_ckpt_label]

            override_text = st.text_area("Edit specialist output for this step", value=sel_node.get("details", {}).get("specialist_output", ""), height=100)

            if st.button("Run Partial Replay from Step", use_container_width=True):
                with st.spinner("Resuming execution from checkpoint with override..."):
                    updates = {"specialist_output": override_text}
                    new_tid, res = partial_replay_task(task_id, sel_node["node_id"], updates, user_id=user_id)
                    st.session_state.replay_result = {"new_task_id": new_tid, "result": res, "mode": "partial", "orig_task_id": task_id}
                    st.success(f"Partial replay complete (New Task ID: `{new_tid[:12]}`).")
                    st.rerun()
        else:
            st.info("No checkpoint steps are available for this task run. Run a fresh task from the Workspace page to enable step-by-step partial replay.")

    # ── Trace Diff Viewer ──────────────────────────────────────────────────────
    rep_info = st.session_state.get("replay_result")
    if rep_info and rep_info.get("orig_task_id") == task_id:
        st.markdown('<div class="section-label">Trace Diff Viewer (Original vs Replay)</div>', unsafe_allow_html=True)

        orig_trace = trace_data
        replay_trace = build_execution_trace(rep_info["new_task_id"], user_id=user_id)
        diff_summary = compare_traces(orig_trace, replay_trace)

        if diff_summary["diverged"]:
            div_p = diff_summary["divergence_point"]
            st.warning(f"⚡ Traces DIVERGED at Step {div_p['step_index']} ({div_p['step_name']}): {div_p['reason']}")
            st.info("💡 **Note on Replay Path Non-Determinism**: Both step SEQUENCE (graph routing path) and step CONTENT can legitimately differ during replay. Because downstream LLM reviewer nodes evaluate fresh specialist outputs autonomously, sampling variations in LLM reviewer decisions (such as 'retry' vs 'approve') can introduce or bypass retry loops, resulting in different node path sequences.")
        else:
            st.success("✓ Replay trace MATCHED the original trace step-by-step.")

        for cmp_item in diff_summary["comparisons"]:
            s_idx = cmp_item["step_index"]
            match_icon = "✓ MATCH" if cmp_item["matched"] else "⚡ DIVERGED"
            st.markdown(f"**Step {s_idx}**: Orig `{cmp_item['original_step']}` vs Replay `{cmp_item['replay_step']}` — **{match_icon}** ({cmp_item['divergence_reason']})")


def render_auth_page() -> None:
    """Render the split-screen authentication (Login / Signup) page (100vh height, zero scrollbar)."""
    st.markdown(
        '''<style>
        /* Zero out main App view container padding and set 100vh viewport height */
        html, body, .stApp {
            background-color: #355E58 !important;
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .auth-container [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stAppViewContainer"],
        .main,
        [data-testid="stMain"],
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }

        /* Horizontal Block fills 100vw width and 100vh height FLUSH at y=0, x=0 */
        div[data-testid="stHorizontalBlock"]:first-of-type,
        [data-testid="stHorizontalBlock"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            min-height: 100vh !important;
            max-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
            gap: 0 !important;
            display: flex !important;
            z-index: 99999 !important;
        }

        /* RESET ALL STREAMLIT COLUMN BACKGROUNDS & PADDING */
        [data-testid="stColumn"],
        [data-testid="column"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* LEFT COLUMN (~52vw width) — Dark Spruce Hero (#355E58) */
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(1),
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(1) {
            width: 52vw !important;
            min-width: 52vw !important;
            max-width: 52vw !important;
            height: 100vh !important;
            background-color: #355E58 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: flex-start !important;
            padding: 4rem 5rem !important;
            box-sizing: border-box !important;
            margin: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(1) > [data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(1) > [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            height: 100% !important;
            width: 100% !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* RIGHT COLUMN (~48vw width) — Solid Lace (#FFEDD1) filling edge-to-edge 100% */
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(2),
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(2) {
            width: 48vw !important;
            min-width: 48vw !important;
            max-width: 48vw !important;
            height: 100vh !important;
            background-color: #FFEDD1 !important; /* Lace background filling entire right panel edge-to-edge! */
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 2rem !important;
            box-sizing: border-box !important;
            margin: 0 !important;
        }

        /* SINGLE WHITE AUTH CARD CONTAINER IN RIGHT COLUMN */
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(2) > [data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(2) > [data-testid="stVerticalBlock"] {
            background-color: #FFFFFF !important;
            border-radius: 14px !important;
            border: 1px solid #EAE3D2 !important;
            box-shadow: 0 10px 30px rgba(53, 94, 88, 0.12) !important;
            padding: 2.2rem 2.2rem 2rem 2.2rem !important;
            width: 100% !important;
            max-width: 420px !important;
            height: fit-content !important;
            min-height: auto !important;
            max-height: 90vh !important;
            flex: none !important;
            margin: auto !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        /* HERO TEXT IN LEFT COLUMN */
        .auth-left-hero {
            width: 100% !important;
            max-width: 480px !important;
        }

        .auth-hero-brand {
            font-size: 0.9rem !important;
            font-weight: 800 !important;
            color: #72B0AB !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            margin-bottom: 1.2rem !important;
        }

        .auth-hero-title {
            font-size: 2.7rem !important;
            font-weight: 850 !important;
            color: #FFFFFF !important;
            line-height: 1.15 !important;
            letter-spacing: -0.03em !important;
            margin: 0 0 1.2rem 0 !important;
            display: block !important;
        }

        .auth-hero-sub {
            font-size: 1.05rem !important;
            font-weight: 400 !important;
            color: #D5E3E0 !important;
            line-height: 1.6 !important;
            margin: 0 !important;
            display: block !important;
        }

        /* Prevent inner forms/columns from adding extra boxes */
        [data-testid="stForm"],
        [data-testid="stColumn"] [data-testid="stColumn"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Logo inside card */
        .auth-card-logo {
            text-align: center;
            margin-bottom: 1.2rem;
        }

        .auth-card-logo-icon {
            font-size: 2.4rem;
            line-height: 1;
            margin-bottom: 0.2rem;
        }

        .auth-card-logo-title {
            font-size: 1.45rem;
            font-weight: 850;
            color: #355E58;
            letter-spacing: -0.02em;
            margin: 0;
        }

        /* REACT ARIA TABS & SELECTION INDICATOR OVERRIDE — SOLID SAPPHIRE (#72B0AB) */
        .react-aria-SelectionIndicator,
        div.react-aria-SelectionIndicator,
        [class*="react-aria-SelectionIndicator"],
        div[class*="SelectionIndicator"],
        html body .stApp [data-testid="stTabs"] div[data-baseweb="tab-highlight"],
        html body .stApp div[data-baseweb="tab-highlight"],
        html body .stApp [data-baseweb="tab-highlight"],
        div[data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
        div[data-baseweb="tab-list"] > div:last-child,
        div[style*="255, 75, 75"],
        div[style*="255,75,75"],
        div[style*="rgb(255"],
        div[style*="FF4B4B"],
        div[style*="ff4b4b"] {
            background-color: #72B0AB !important;
            background: #72B0AB !important;
            border-color: #72B0AB !important;
            height: 3px !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        /* React Aria Tab Buttons */
        div[data-testid="stTab"],
        [data-baseweb="tab"],
        [data-testid="stTabs"] [data-baseweb="tab"],
        button[data-baseweb="tab"] {
            flex: 1 !important;
            height: 38px !important;
            border-radius: 26px !important;
            border: 1px solid transparent !important;
            border-bottom: none !important;
            outline: none !important;
            background-color: transparent !important;
            color: #355E58 !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            padding: 0 !important;
            justify-content: center !important;
            align-items: center !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }

        /* React Aria Active Selected Tab */
        div[data-testid="stTab"][aria-selected="true"],
        [data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #72B0AB !important;
            background: #72B0AB !important;
            color: #FFFFFF !important;
            border-radius: 26px !important;
            border: 1px solid #72B0AB !important;
            box-shadow: 0 2px 6px rgba(114, 176, 171, 0.3) !important;
        }

        div[data-testid="stTab"][aria-selected="true"] *,
        [data-baseweb="tab"][aria-selected="true"] *,
        button[data-baseweb="tab"][aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        html body .stApp div[data-baseweb="tab-border"],
        html body .stApp [data-baseweb="tab-border"] {
            background-color: #E0EBE8 !important;
        }

        /* Remove any default red focus ring / border on tab buttons */
        [data-baseweb="tab"]:focus,
        button[data-baseweb="tab"]:focus,
        [data-baseweb="tab"]:focus-visible,
        button[data-baseweb="tab"]:focus-visible {
            outline: 2px solid #72B0AB !important;
            outline-offset: -2px !important;
            box-shadow: none !important;
            border-color: #72B0AB !important;
        }

        /* Active Selected Pill */
        [data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #72B0AB !important;
            background: #72B0AB !important;
            color: #FFFFFF !important;
            border-radius: 26px !important;
            border: 1px solid #72B0AB !important;
            box-shadow: 0 2px 6px rgba(114, 176, 171, 0.3) !important;
        }

        [data-baseweb="tab"][aria-selected="true"] *,
        button[data-baseweb="tab"][aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Input Field Styling */
        [data-testid="stTextInput"] {
            margin-bottom: 0.75rem !important;
        }

        [data-testid="stTextInput"] label p {
            color: #355E58 !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stTextInput"] input {
            background-color: #FFFFFF !important;
            border: 1px solid #D5E3E0 !important;
            border-radius: 8px !important;
            padding: 0.55rem 0.85rem !important;
            color: #355E58 !important;
            font-size: 0.9rem !important;
            width: 100% !important;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: #72B0AB !important;
            box-shadow: 0 0 0 3px rgba(114, 176, 171, 0.2) !important;
            outline: none !important;
        }

        /* PASSWORD EYE TOGGLE BUTTON TRANSPARENT FIX */
        button[aria-label="Show password"],
        button[aria-label="Hide password"],
        button[aria-label*="password"],
        [data-testid="stInputPasswordToggle"] button,
        [data-testid="stInputPasswordToggle"],
        div[data-baseweb="input"] > div:last-child {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-baseweb="input"] svg,
        [data-testid="stInputPasswordToggle"] svg,
        button[aria-label*="password"] svg {
            fill: #72B0AB !important;
            color: #72B0AB !important;
            stroke: #72B0AB !important;
        }

        /* PRIMARY SUBMIT BUTTON — SOLID SAPPHIRE (#72B0AB) */
        [data-testid="stFormSubmitButton"] button,
        button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        .stButton > button {
            background-color: #72B0AB !important;
            background: #72B0AB !important;
            color: #FFFFFF !important;
            border: 1px solid #72B0AB !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 0.65rem 1rem !important;
            width: 100% !important;
            transition: background-color 0.2s ease !important;
            margin-top: 0.5rem !important;
            box-shadow: 0 3px 10px rgba(114, 176, 171, 0.25) !important;
        }

        [data-testid="stFormSubmitButton"] button:hover,
        button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        .stButton > button:hover {
            background-color: #5B9590 !important;
            background: #5B9590 !important;
            border-color: #5B9590 !important;
            color: #FFFFFF !important;
        }

        [data-testid="stFormSubmitButton"] button p,
        button[kind="primary"] p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* MOBILE RESPONSIVE STACKING (<768px) */
        @media (max-width: 768px) {
            html, body, .stApp, [data-testid="stAppViewContainer"], .main {
                height: auto !important;
                max-height: none !important;
                overflow-y: auto !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type {
                flex-direction: column !important;
                height: auto !important;
                min-height: auto !important;
                max-height: none !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(1),
            div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(1) {
                width: 100% !important;
                height: auto !important;
                padding: 3rem 2rem !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-of-type(2),
            div[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(2) {
                width: 100% !important;
                height: auto !important;
                padding: 3rem 1.5rem !important;
            }
        }
        </style>''',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown(
            '''<div class="auth-left-hero">
                <div class="auth-hero-brand">🐝 AGENTHIVE</div>
                <div class="auth-hero-title">Welcome to AgentHive.</div>
                <div class="auth-hero-sub">Sign in to your account to manage specialized multi-agent workflows, review real-time tasks, and inspect execution traces.</div>
            </div>''',
            unsafe_allow_html=True,
        )

    with right_col:
        # 1. Logo Header at top of card
        st.markdown(
            '''<div class="auth-card-logo">
                <div class="auth-card-logo-icon">🐝</div>
                <div class="auth-card-logo-title">AgentHive</div>
                <div class="auth-card-logo-sub">ORCHESTRATED INTELLIGENCE</div>
            </div>''',
            unsafe_allow_html=True,
        )

        # 2. Pill Tabs inside same card
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", value="", placeholder="alex_dev", key="login_username_input").strip()
                password = st.text_input("Password", type="password", value="", placeholder="••••••••", key="login_password_input")
                submit_login = st.form_submit_button("Log In", type="primary", use_container_width=True)

                if submit_login:
                    success, msg = verify_user(username, password)
                    if success:
                        clean_user = username.strip().lower()
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"] = clean_user
                        st.session_state["username"] = clean_user
                        st.session_state["active_user_id"] = clean_user
                        st.success("Login successful! Loading workspace...")
                        st.rerun()
                    else:
                        st.error(msg)
                        st.info("💡 **Tip**: If you haven't created your account yet, click the **Sign Up** tab above to create an account.")

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                new_username = st.text_input("Choose Username", value="", placeholder="alex_dev", key="signup_username_input").strip()
                new_password = st.text_input("Choose Password", type="password", value="", placeholder="••••••••", key="signup_password_input")
                confirm_password = st.text_input("Confirm Password", type="password", value="", placeholder="••••••••", key="signup_confirm_password_input")
                submit_signup = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submit_signup:
                    if new_password != confirm_password:
                        st.error("Passwords do not match. Please try again.")
                    else:
                        success, msg = create_user(new_username, new_password)
                        if success:
                            clean_user = new_username.strip().lower()
                            st.session_state["logged_in"] = True
                            st.session_state["user_id"] = clean_user
                            st.session_state["username"] = clean_user
                            st.session_state["active_user_id"] = clean_user
                            st.success("Account created successfully! Loading workspace...")
                            st.rerun()
                        else:
                            st.error(msg)


def render_sidebar(available: bool) -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><div class="sidebar-brand-name">🐝 AgentHive</div><div class="sidebar-caption">ORCHESTRATED INTELLIGENCE</div></div>', unsafe_allow_html=True)

        if st.session_state.get("logged_in"):
            user_display = st.session_state.get("username", st.session_state.get("user_id", "User"))
            st.markdown(
                f'''<div class="user-status-pill">
                    <span class="status-dot">●</span>
                    <span class="status-text">Logged in as <strong>{html.escape(str(user_display))}</strong></span>
                </div>''',
                unsafe_allow_html=True,
            )

        pending_count = 0
        try:
            from src.orchestration.approval import get_pending_approvals
            active_user = st.session_state.get("user_id", "default")
            pending_count = len(get_pending_approvals(active_user))
        except Exception:
            pass

        queue_label = f"Approval Queue ({pending_count})" if pending_count > 0 else "Approval Queue"
        options = ["Dashboard", "Workspace", "Task History", "Memory Dashboard", "Trace Explorer", queue_label]

        page = st.radio("Navigation", options, key="page_navigation", label_visibility="collapsed")
        clean_page = "Approval Queue" if "Approval Queue" in str(page) else str(page)

        st.divider()
        st.markdown("##### SYSTEM STATUS")
        status = "● Connected" if available else "● Offline"
        color = "#72B0AB" if available else "#BCDDDC"
        st.markdown(f'<div class="system-chip" style="color:{color}">{status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="system-chip">Model · {html.escape(settings.ollama_model)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="system-chip">Memory · Local persistent store</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("##### WORKFLOW")
        st.caption("Recall context\n\nRoute task\n\nGenerate response\n\nReview and save")
        st.divider()

        if st.session_state.get("logged_in"):
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["user_id"] = None
                st.session_state["username"] = None
                st.session_state["active_user_id"] = None
                st.rerun()

    return clean_page


def render_app() -> None:
    """Configure and render the AgentHive application."""
    settings.ensure_runtime_directories()
    st.set_page_config(page_title="AgentHive | Multi-Agent Workspace", page_icon="🐝", layout="wide", initial_sidebar_state="expanded")
    apply_custom_styles()

    if not st.session_state.get("logged_in"):
        render_auth_page()
        return

    available = OllamaClient().is_available()
    page = render_sidebar(available)
    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Workspace":
        render_workspace_page()
    elif page == "Memory Dashboard":
        render_memory_dashboard_page()
    elif page == "Trace Explorer":
        render_trace_explorer_page()
    elif page == "Approval Queue":
        render_approval_queue_page()
    else:
        render_history_page()


def render_workspace_page() -> None:
    st.markdown('<div style="padding-left: 0.35rem; margin-left: 0;">', unsafe_allow_html=True)
    st.title("Agent workspace")
    st.caption("Write a clear objective. AgentHive will take care of delegation, review, and memory.")

    # User scoping field
    user_id = st.session_state.get("user_id") or st.session_state.get("active_user_id", "default")
    st.session_state.active_user_id = user_id
    st.markdown(f'<div class="system-chip" style="margin-bottom: 16px;">Active Identity · <b>{html.escape(user_id)}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # A form submits the text field and button click together, so one click runs the task.
    with st.form("agent_task_form", clear_on_submit=False, border=True):
        task = st.text_area(
            "Task objective",
            placeholder="Example: Compare REST and GraphQL for a mobile application, including when to choose each approach.",
            height=175,
            key="task_input",
            label_visibility="visible",
        )
        pause_step_by_step = st.checkbox("Pause before each step (Step-by-step human review)", key="pause_step_input")
        left, right = st.columns([1, 5])
        with left:
            run = st.form_submit_button("Run task", type="primary", use_container_width=True)
        with right:
            st.caption(f"The workflow runs for user '{user_id}'. Results are saved to memory and local task history.")

    if run and not task.strip():
        st.warning("Please enter a task objective before running the workflow.")
        return

    if run:
        start = time.perf_counter()
        from uuid import uuid4
        task_id = str(uuid4())
        pause_mode = "step_by_step" if pause_step_by_step else "normal"

        try:
            with st.status("Agents are collaborating…", expanded=True) as progress:
                st.write("Recalling relevant context")
                config = {"configurable": {"thread_id": task_id}}
                result = build_graph().invoke(
                    {"task": task.strip(), "user_id": user_id, "task_id": task_id, "pause_mode": pause_mode, "trace": []},
                    config=config,
                )
                st.write("Reviewing the completed response")
                progress.update(label="Workflow executed", state="complete", expanded=False)
            elapsed = time.perf_counter() - start
            st.session_state.latest_result = {"result": result, "elapsed": elapsed}

            history = load_task_history()
            history.append({
                "timestamp": datetime.now().strftime("%d %b %Y · %I:%M %p"),
                "task": task.strip(),
                "user_id": user_id,
                "task_id": task_id,
                "selected_agent": result.get("selected_agent", "research") if isinstance(result, dict) else "research",
                "trace": result.get("trace", []) if isinstance(result, dict) else [],
                "final_answer": result.get("final_answer", result.get("specialist_output", "")) if isinstance(result, dict) else "",
                "runtime_seconds": round(elapsed, 2),
            })
            save_task_history(history[-100:])
        except Exception as error:
            st.error("AgentHive could not complete this task. Confirm that Ollama is running and the configured model is installed.")
            with st.expander("Technical details"):
                st.code(str(error))

    latest = st.session_state.get("latest_result")
    if latest:
        res = latest["result"]
        # Ensure 'paused for approval' banner and the run card are mutually exclusive
        is_paused = isinstance(res, dict) and (res.get("__interrupt__") or res.get("workflow_status") in ("paused_for_approval", "paused"))
        if is_paused:
            st.warning("Task execution is PAUSED for human approval. Navigate to the Approval Queue page to review and resume.")
        else:
            show_result(res, latest.get("elapsed"))


def derive_ui_flags(result: dict[str, Any]) -> dict[str, bool]:
    """Derive UI flags (show_paused_banner vs show_completed_card) mutually exclusively."""
    if not isinstance(result, dict):
        return {"show_paused_banner": False, "show_completed_card": False}

    is_paused = bool(result.get("is_paused") or result.get("__interrupt__") or result.get("workflow_status") in ("paused_for_approval", "paused"))
    if is_paused:
        return {"show_paused_banner": True, "show_completed_card": False}

    return {"show_paused_banner": False, "show_completed_card": True}


def _extract_answer_summary_ui(text: str) -> str:
    marker = "Answer summary:"
    idx = text.find(marker)
    if idx != -1:
        snippet = text[idx + len(marker):].strip()
    else:
        snippet = text.strip()
    if len(snippet) > 200:
        snippet = snippet[:200].rstrip() + "…"
    return snippet


def render_memory_dashboard_page() -> None:
    st.title("Memory Dashboard")
    st.caption("Inspect, manage, and optimize user-scoped long-term memories.")

    user_id = st.session_state.get("user_id") or st.session_state.get("active_user_id", "default")
    st.session_state.active_user_id = user_id
    st.markdown(f'<div class="system-chip" style="margin-bottom: 16px;">Active Identity · <b>{html.escape(user_id)}</b></div>', unsafe_allow_html=True)

    try:
        from src.tools.memory_tool import AgentMemoryTool
        memory_tool = AgentMemoryTool()
    except Exception as exc:
        st.warning("Memory System is operating in fallback mode (ChromaDB or Ollama unavailable).")
        st.caption(f"Error detail: {exc}")
        return

    stats = memory_tool.get_stats(user_id)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_stat_card("Stored memories", stats.get("total_memories", 0), f"User: {user_id}")
    with m2:
        render_stat_card("Avg importance", f"{stats.get('avg_importance', 0.0):.2f}", "Scored 0.0 - 1.0")
    with m3:
        render_stat_card("Success / Failure", f"{stats.get('success_count', 0)} / {stats.get('failure_count', 0)}", "Recorded outcomes")
    with m4:
        render_stat_card("Most used agent", stats.get("most_used_agent", "—"), "Primary specialist")

    st.markdown('<div class="section-label">Memory Management Actions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Consolidate Duplicates", use_container_width=True):
            count = memory_tool.consolidate(user_id)
            st.success(f"Consolidated {count} duplicate memory entries.")
            st.rerun()
    with c2:
        if st.button("Expire Low-Importance", use_container_width=True):
            count = memory_tool.expire_old_memories(user_id)
            st.info(f"Expired {count} low-importance memories.")
            st.rerun()
    with c3:
        if st.button("Clear All My Memory", type="primary", use_container_width=True):
            count = memory_tool.delete_all(user_id)
            st.warning(f"Cleared all {count} memories for user '{user_id}'.")
            st.rerun()

    st.markdown('<div class="section-label">Stored Memories</div>', unsafe_allow_html=True)
    memories = memory_tool.list_memories(user_id, limit=100)

    if not memories:
        st.info(f"No stored memories found for user '{user_id}'. Complete tasks in the workspace to save memories.")
        return

    for mem in memories:
        mem_id = mem["id"]
        meta = mem.get("metadata", {})
        task_type = meta.get("task_type", "unknown")
        importance = float(meta.get("importance_score", 0.5))
        timestamp = meta.get("timestamp", "Unknown date")
        if len(timestamp) > 10:
            timestamp = timestamp[:10] + " " + timestamp[11:16]

        text = mem.get("text", "")
        answer_summary = _extract_answer_summary_ui(text)

        with st.container(border=True):
            col_a, col_b, col_c = st.columns([4, 2, 1])
            with col_a:
                st.markdown(f"**[{task_type.upper()}] Memory Entry**")
                # Render formatted output so code/Python # comments render in st.code() instead of raw Markdown H1 headers
                render_formatted_output(answer_summary)
                if len(text) > 200:
                    with st.expander("🔍 View Full Stored Memory Content"):
                        render_formatted_output(text)
                st.caption(f"Agent: {meta.get('selected_agent', 'unknown')} | Outcome: {meta.get('outcome', 'unknown')} | Tools: {meta.get('tools_used', 'none')}")
            with col_b:
                st.markdown(f"**Importance Score**: `{importance:.3f}`")
                st.caption(f"Date: {timestamp}")
            with col_c:
                if st.button("Delete", key=f"del_{mem_id}", use_container_width=True):
                    memory_tool.delete_memory(mem_id, user_id)
                    st.toast(f"Deleted memory entry {mem_id[:8]}")
                    st.rerun()


def render_approval_queue_page() -> None:
    st.title("Approval Queue")
    st.caption("Review paused tasks, inspect proposed actions, ask clarifying questions, and approve or take over execution.")

    user_id = st.session_state.get("user_id") or st.session_state.get("active_user_id", "default")
    st.session_state.active_user_id = user_id
    st.markdown(f'<div class="system-chip" style="margin-bottom: 16px;">Active Identity · <b>{html.escape(user_id)}</b></div>', unsafe_allow_html=True)

    from src.orchestration.approval import get_pending_approvals, clear_pending_approvals
    pending_items = get_pending_approvals(user_id)

    if not pending_items:
        st.info(f"No pending approvals found for user '{user_id}'. All workflow pipelines are running smoothly.")
        return

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f'<div class="section-label">Pending Approval Items ({len(pending_items)})</div>', unsafe_allow_html=True)
    with col_h2:
        if st.button("Clear All Pending", key="clear_all_pending_queue"):
            count = clear_pending_approvals(user_id)
            st.toast(f"Cleared {count} pending approval(s).")
            st.rerun()

    task_options = {f"Task {item['task_id'][:8]} — {item['task'][:50]}...": item for item in pending_items}
    selected_label = st.selectbox("Select pending task to review", list(task_options.keys()))
    selected_item = task_options[selected_label]

    task_id = selected_item["task_id"]
    trigger_reason = selected_item["trigger_reason"]
    approval_level = selected_item["approval_level"]
    proposed_action = selected_item["proposed_action"]
    reasoning = selected_item.get("reasoning", "")
    recalled_memories = selected_item.get("recalled_memories", "")

    st.markdown('<div class="result-row-gap"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.7, 1])

    with c1:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Task Context & Objective</div>', unsafe_allow_html=True)
            st.markdown(f"**Task**: {selected_item['task']}")
            st.caption(f"Task ID: `{task_id}` | User: `{user_id}` | Queued At: `{selected_item['timestamp'][:16]}`")
            st.divider()
            st.markdown(f"**Trigger Reason**: `{trigger_reason}`")
            st.markdown(f"**Default Approval Level**: `{approval_level}`")
            st.markdown(f"**Proposed Action**: {proposed_action}")
            if reasoning:
                st.markdown(f"**Instruction / Reasoning**: {reasoning}")

        # Clarifying Chat Panel
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Interactive Clarifying Chat</div>', unsafe_allow_html=True)
            st.caption("Ask the AI Assistant clarifying questions about this proposed action before deciding.")

            question_key = f"chat_q_{task_id}"
            user_question = st.text_input("Your question", placeholder="Example: What are the security risks or side-effects of running this tool call?", key=question_key)
            if st.button("Ask Assistant", type="secondary", key=f"btn_ask_{task_id}") and user_question.strip():
                with st.spinner("Assistant is analyzing..."):
                    llm = OllamaClient()
                    chat_prompt = (
                        f"Task Objective: {selected_item['task']}\n"
                        f"Proposed Action: {proposed_action}\n"
                        f"Reasoning: {reasoning}\n\n"
                        f"User's Question: {user_question.strip()}\n\n"
                        "Provide a concise, helpful explanation of the risks, side effects, or implications of this proposed action."
                    )
                    ans = llm.ask(chat_prompt, "You are AgentHive's Human-in-the-Loop review assistant.")
                    st.session_state[f"chat_ans_{task_id}"] = ans

            chat_response = st.session_state.get(f"chat_ans_{task_id}")
            if chat_response:
                st.markdown(f"**Assistant Response**:\n{chat_response}")

    with c2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Recalled Memories</div>', unsafe_allow_html=True)
            if recalled_memories:
                st.markdown(recalled_memories)
            else:
                st.caption("No relevant past memories recalled for this task.")

    # Action Buttons: Granular Approval Levels
    st.markdown('<div class="section-label">Approval Decisions (Resume Workflow)</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("Notify Only", help="Log approval event and resume without pausing future steps", use_container_width=True):
            _resume_task(task_id, {"action": "notify", "decision": "approve"})

    with b2:
        if st.button("Approve Action", type="primary", help="Approve just this specific tool call / subtask", use_container_width=True):
            _resume_task(task_id, {"action": "approve_action", "decision": "approve"})

    with b3:
        if st.button("Approve Plan", help="Approve all remaining subtasks without further pauses", use_container_width=True):
            _resume_task(task_id, {"action": "approve_plan", "decision": "approve"})

    with b4:
        with st.popover("Take Over"):
            st.markdown("**Take Over Action Output**")
            st.caption("Supply custom text output for this step, bypassing specialist LLM execution.")
            custom_text = st.text_area("Custom output text", height=120, key=f"takeover_{task_id}")
            if st.button("Submit Manual Output", type="primary", key=f"btn_takeover_{task_id}"):
                _resume_task(task_id, {"action": "take_over", "decision": "approve", "custom_output": custom_text})


def _resume_task(task_id: str, resume_data: dict) -> None:
    """Resume a paused graph using LangGraph Command(resume=...)."""
    try:
        from src.orchestration.graph import build_graph
        graph = build_graph()
        config = {"configurable": {"thread_id": task_id}}
        result = graph.invoke(Command(resume=resume_data), config=config)
        st.session_state.latest_result = {"result": result, "elapsed": None}
        st.success("Workflow resumed successfully.")
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to resume task: {exc}")


def render_history_page() -> None:
    st.title("Task history")
    st.caption("A local record of your recent completed AgentHive runs.")
    history = load_task_history()
    top, actions = st.columns([4, 2])
    with top:
        render_stat_card("Saved runs", len(history), "Available in local history")
    with actions:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Export JSON", json.dumps(history, indent=2, ensure_ascii=False), "agenthive_history.json", "application/json", use_container_width=True)
        with d2:
            if st.button("Clear history", use_container_width=True, disabled=not history):
                save_task_history([])
                st.session_state.pop("latest_result", None)
                st.rerun()

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    if not history:
        st.info("No completed tasks yet. Your first hive run will appear here.")
        return

    for idx, item in enumerate(reversed(history)):
        agent, _, _ = route_label(item.get("selected_agent", "research"))
        task = item.get("task", "Untitled task")
        task_id = item.get("task_id") or f"hist_{idx}_{item.get('timestamp', '')}"
        
        c_task, c_delete = st.columns([5, 1])
        with c_task:
            with st.expander(f"{agent}  ·  {item.get('timestamp', 'Unknown time')}  ·  {task[:70]}"):
                st.markdown("**Objective**")
                st.write(task)
                st.markdown("**Final response**")
                st.markdown(item.get("final_answer", "No saved response."))
                st.markdown("**Workflow trace**")
                render_trace(item.get("trace", []))
        with c_delete:
            if st.button("🗑️ Delete", key=f"del_hist_{task_id}", use_container_width=True, help="Delete this task history entry"):
                target_id = item.get("task_id")
                target_ts = item.get("timestamp")
                new_history = [
                    h for h in history
                    if not ((target_id and h.get("task_id") == target_id) or (target_ts and h.get("timestamp") == target_ts and h.get("task") == task))
                ]
                save_task_history(new_history)
                st.toast("Deleted task history entry.")
                st.rerun()
