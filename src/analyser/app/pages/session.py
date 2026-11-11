"""Session page: settings, datasets, save and reload, and what is registered."""

from __future__ import annotations

import datetime as _dt

import streamlit as st

from analyser.app import state
from analyser.core.dataset import TIME_UNITS
from analyser.core.registry import list_registered
from analyser.core.session import QUARTILE_METHODS, Session
from analyser.io.profiles import analyser_home

S = state.session()


def page() -> None:
    st.title("Session")

    st.subheader("Settings")
    with st.form("ses_settings"):
        c = st.columns(5)
        alpha = c[0].number_input("α", min_value=0.001, max_value=0.5, value=S.settings.alpha, step=0.01, format="%g")
        qm = c[1].selectbox("Quartile method", list(QUARTILE_METHODS),
                            index=list(QUARTILE_METHODS).index(S.settings.quartile_method),
                            help="QUARTILE.INC is Excel's inclusive method (numpy 'linear'), the default.")
        seed = c[2].number_input("Random seed", min_value=0, value=int(S.settings.seed), step=1)
        units = list(TIME_UNITS)
        tu = c[3].selectbox("Time unit", units, index=units.index(S.settings.time_unit))
        cur = c[4].text_input("Currency", value=S.settings.currency, max_chars=5)
        if st.form_submit_button("Save settings"):
            try:
                S.update_settings(alpha=float(alpha), quartile_method=qm, seed=int(seed), time_unit=tu, currency=cur)
                st.success("Settings saved. They apply to new analyses and imports.")
            except ValueError as exc:
                st.error(str(exc))

    st.subheader("Datasets")
    if S.datasets:
        st.dataframe([{"dataset": n, "rows": len(d), "columns": d.data.shape[1], "time unit": d.time_unit,
                       "currency": d.currency, "log steps": len(d.log)} for n, d in S.datasets.items()],
                     hide_index=True, width="stretch")
        drop = st.selectbox("Remove a dataset", ["(keep all)"] + list(S.datasets), key="ses_drop")
        if drop != "(keep all)" and st.button(f"Remove {drop}", key="ses_drop_btn"):
            S.datasets.pop(drop, None)
            st.rerun()
    else:
        st.caption("No datasets yet.")

    st.subheader("Save and reload")
    c = st.columns(2)
    with c[0]:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button("Download session (.json)", S.to_json(), file_name=f"analyser-session-{stamp}.json",
                           mime="application/json", key="ses_download",
                           help="Settings, all results and the datasets, so the analysis can be reopened.")
        folder = analyser_home() / "sessions"
        name = st.text_input("Or save on this computer as", value=f"session-{stamp}", key="ses_name")
        if st.button("Save to folder", key="ses_save"):
            folder.mkdir(parents=True, exist_ok=True)
            path = S.save(folder / f"{name}.json")
            st.success(f"Saved to {path}")
    with c[1]:
        up = st.file_uploader("Reload a session file", type=["json"], key="ses_upload")
        if up is not None and st.button("Replace the current session", key="ses_replace"):
            try:
                state.replace_session(Session.from_json(up.getvalue().decode("utf-8")))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Not a session file: {exc}")
        saved = sorted((analyser_home() / "sessions").glob("*.json")) if (analyser_home() / "sessions").exists() else []
        if saved:
            pick = st.selectbox("Saved sessions", [p.name for p in saved], key="ses_saved")
            if st.button("Open saved session", key="ses_open"):
                state.replace_session(Session.load(analyser_home() / "sessions" / pick))
                st.rerun()

    st.subheader("Registered methods")
    st.caption("Everything the engine knows. Menus, the guided selector (phase 6) and docs read this list.")
    reg = list_registered(as_frame=True)
    cols = [c for c in ["kind", "id", "name", "family", "samples", "data", "parameter", "assumptions", "sources"]
            if c in reg.columns]
    st.dataframe(reg[cols].astype(str).replace({"nan": "", "None": "", "()": ""}), hide_index=True, width="stretch")


page()
