"""Results page: the session's result history; renders any result object, including saved JSON files."""

from __future__ import annotations

import streamlit as st

from analyser.app import state
from analyser.app.render import render_result
from analyser.core.results import result_from_json

S = state.session()


def page() -> None:
    st.title("Results")

    with st.expander("Open a saved result (.json)"):
        up = st.file_uploader("Result file", type=["json"], key="res_upload")
        if up is not None:
            try:
                loaded = result_from_json(up.getvalue().decode("utf-8"))
                st.success(f"Loaded: {loaded.summary()}")
                if st.button("Add to this session", key="res_add"):
                    if all(r.id != loaded.id for r in S.results):
                        S.results.append(loaded)
                    st.session_state["res_pick"] = loaded.id
                    st.rerun()
                render_result(loaded, key="res_loaded")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Not a result file: {exc}")

    if not S.results:
        st.info("No results yet. Run a test or a fit on the Analyse page.")
        st.page_link("pages/analyse.py", label="Go to Analyse", icon=":material/functions:")
        return

    st.dataframe(S.history().drop(columns=["id"]), hide_index=True, width="stretch")
    ids = [r.id for r in S.results]
    labels = {r.id: f"#{i + 1}  {r.summary()}" for i, r in enumerate(S.results)}
    default = st.session_state.get("res_pick")
    index = ids.index(default) if default in ids else len(ids) - 1
    pick = st.selectbox("Show result", ids, index=index, format_func=labels.get, key="res_select")
    render_result(S.get(pick), key="res_main")
    if st.button("Remove this result from the session", key="res_remove"):
        S.remove(pick)
        st.session_state.pop("res_pick", None)
        st.rerun()


page()
