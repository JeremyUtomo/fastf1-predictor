"""Shared look and feel: CSS on top of the dark theme in .streamlit/config.toml."""
from __future__ import annotations

import html

import streamlit as st

MUTED = "#8B93A1"
BORDER = "#262B34"
SURFACE = "#161A21"

_CSS = f"""
<style>
  .block-container {{padding-top: 3.25rem; padding-bottom: 4rem; max-width: 1480px;}}
  header[data-testid="stHeader"] {{background: transparent;}}

  /* section labels: markdown h5 renders as a small uppercase label */
  h5 {{text-transform: uppercase; letter-spacing: .08em; color: {MUTED} !important;
       font-weight: 600 !important; margin: 1.25rem 0 .35rem 0 !important; padding: 0 !important;}}
  h1, h2, h3, h4 {{letter-spacing: -0.01em;}}

  /* metrics as compact cards */
  [data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid {BORDER};
       border-radius: 6px; padding: 10px 14px 8px 14px;}}
  [data-testid="stMetricLabel"] p {{text-transform: uppercase; letter-spacing: .06em;
       font-size: .7rem !important; color: {MUTED};}}
  [data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums;}}
  [data-testid="stMetricDelta"] {{font-size: .75rem;}}
  [data-testid="stMetricDelta"] > div {{background: transparent !important; padding-left: 0 !important;
       color: {MUTED} !important;}}

  /* tabs: quiet underline style */
  .stTabs [data-baseweb="tab-list"] {{gap: 2px; border-bottom: 1px solid {BORDER};}}
  .stTabs [data-baseweb="tab"] {{padding: 6px 14px; font-weight: 500; color: {MUTED};}}
  .stTabs [aria-selected="true"] {{color: #E4E7EC;}}

  /* expanders and alerts: flatter */
  [data-testid="stExpander"] details {{border-color: {BORDER};}}
  [data-testid="stExpander"] summary p {{font-weight: 500;}}
  [data-testid="stAlert"] {{border-radius: 6px;}}

  /* numbers line up in tables */
  [data-testid="stDataFrame"] {{font-variant-numeric: tabular-nums;}}

  /* page header */
  .app-header {{display: flex; align-items: baseline; gap: 14px; padding-bottom: 10px;
       border-bottom: 1px solid {BORDER}; margin-bottom: 6px;}}
  .app-header .title {{font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em;}}
  .app-header .context {{color: {MUTED}; font-size: .9rem;}}
  .app-header .dot {{width: 8px; height: 8px; border-radius: 2px; background: #E10600;
       display: inline-block; transform: translateY(-2px);}}
  .sidebar-brand {{font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em;
       display: flex; align-items: center; gap: 8px;}}
  .sidebar-brand .dot {{width: 8px; height: 8px; border-radius: 2px; background: #E10600;}}
  .sidebar-sub {{color: {MUTED}; font-size: .8rem; margin: 2px 0 14px 0;}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(context: str = "", target=None) -> None:
    (target or st).markdown(
        f'<div class="app-header"><span class="dot"></span><span class="title">FastF1 Explorer'
        f'</span><span class="context">{html.escape(context)}</span></div>',
        unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="sidebar-brand"><span class="dot"></span>FastF1 Explorer</div>'
        '<div class="sidebar-sub">Formula 1 timing, telemetry and predictions</div>',
        unsafe_allow_html=True)
