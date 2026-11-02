"""Streamlit dashboard for the MT Trading Swarm (``streamlit run dashboard/app.py``).

A thin client of the REST API: :mod:`dashboard.client` (typed HTTP client),
:mod:`dashboard.charts` (plotly figures), :mod:`dashboard.formatting` (tables and number
formats) and :mod:`dashboard.app` (the Streamlit pages). It does not import the trading
engine, so it can run in its own container pointed at ``API_URL``.
"""
