"""Data Analyser, local app. Start it with:  python -m analyser"""

from __future__ import annotations

import streamlit as st

from analyser.app import state

st.set_page_config(page_title="Data Analyser", page_icon=":material/query_stats:", layout="wide")

pages = [
    st.Page("pages/import_data.py", title="Import data", icon=":material/upload_file:", default=True),
    st.Page("pages/analyse.py", title="Analyse", icon=":material/functions:"),
    st.Page("pages/results.py", title="Results", icon=":material/fact_check:"),
    st.Page("pages/session.py", title="Session", icon=":material/tune:"),
]
nav = st.navigation(pages)
nav.run()

# The sidebar summary is drawn after the page, so it includes what the page just added.
s = state.session()
with st.sidebar:
    st.caption("Session")
    st.markdown(
        f"α = {s.settings.alpha:g} · {s.settings.time_unit} · {s.settings.currency}  \n"
        f"{len(s.datasets)} dataset(s) · {len(s.results)} result(s)"
    )
    active = state.active_dataset()
    if active is not None:
        st.caption(f"Active dataset: **{active.name}** ({len(active)} rows)")
