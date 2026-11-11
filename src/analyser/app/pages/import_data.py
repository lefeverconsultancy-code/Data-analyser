"""Import page: read a file or pasted table, map columns to roles, derive measures, check the data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyser.app import state
from analyser.app.render import STATUS
from analyser.core.dataset import AMOUNT_KINDS, ROLE_LABELS, TIME_UNITS, example_value
from analyser.io.clean import quality_checks
from analyser.io.derive import (
    available_derivations,
    derive,
    period_table,
    picks_per_resource_hour,
    profile_table,
)
from analyser.io.load import EXCEL_SUFFIXES, list_sheets, load
from analyser.io.mapping import ColumnMapping, apply_mapping
from analyser.io.profiles import apply_profile, list_profiles, profile_from_dataset, save_profile

S = state.session()


def page() -> None:
    LABEL_TO_ROLE = {label: role for role, label in ROLE_LABELS.items()}

    st.title("Import data")
    st.caption("Read a CSV or Excel export, or paste a table copied from Excel. Dutch and English "
               "formats are recognised: decimal comma or point, day-first dates.")

    # ---------------------------------------------------------------- 1. Read
    st.subheader("1. Read")
    kind = st.radio("Source", ["Upload a file", "Paste a table", "Sample order data"], horizontal=True,
                    key="imp_kind")
    source, label, is_excel, default_header = None, None, False, 1
    if kind == "Upload a file":
        up = st.file_uploader("CSV or Excel file", type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls"],
                              key="imp_upload")
        if up is not None:
            source, label = up, up.name
            is_excel = up.name.lower().endswith(EXCEL_SUFFIXES)
    elif kind == "Paste a table":
        text = st.text_area("Paste cells (tab-separated from Excel) or CSV text, with a header row",
                            height=160, key="imp_paste")
        if text.strip():
            source, label = text, "pasted table"
    else:
        files = state.example_files()
        if not files:
            st.warning("The examples folder was not found. Run `python examples/make_examples.py` in the repository.")
        else:
            choice = st.selectbox("Sample file", list(files), key="imp_example")
            source, label = str(files[choice]), choice
            is_excel = choice.endswith(".xlsx")
            default_header = 4 if is_excel else 1

    with st.expander("Read options", expanded=is_excel):
        c = st.columns(4)
        sheet = None
        if is_excel and source is not None:
            try:
                sheet = c[0].selectbox("Sheet", list_sheets(source), key=f"imp_sheet_{label}")
            except Exception as exc:  # noqa: BLE001 - show any read error to the user
                st.error(f"Could not open the workbook: {exc}")
        header_row = c[1].number_input("Header row (1 = first row)", min_value=1, value=default_header,
                                       key=f"imp_header_{label}")
        separator = None
        if not is_excel:
            sep_label = c[2].selectbox("Separator", ["auto", ";", ",", "tab", "|"], key="imp_sep")
            separator = None if sep_label == "auto" else ("\t" if sep_label == "tab" else sep_label)
        dec_label = c[3].selectbox("Decimal separator", ["auto", "comma (12,5)", "point (12.5)"], key="imp_dec")
        decimal = {"auto": None, "comma (12,5)": ",", "point (12.5)": "."}[dec_label]
        c2 = st.columns(4)
        date_label = c2[0].selectbox("Date order", ["auto (day-first if unclear)", "day-first (31-12-2026)",
                                                    "month-first (12/31/2026)"], key="imp_dates")
        dayfirst = {"auto (day-first if unclear)": None, "day-first (31-12-2026)": True,
                    "month-first (12/31/2026)": False}[date_label]
        units = list(TIME_UNITS)
        time_unit = c2[1].selectbox("Time unit", units, index=units.index(S.settings.time_unit), key="imp_unit")
        currency = c2[2].text_input("Currency", value=S.settings.currency, max_chars=5, key="imp_cur")
        ds_name = c2[3].text_input("Dataset name", value=(label or "dataset").rsplit(".", 1)[0], key=f"imp_name_{label}")

    profiles = list_profiles()
    profile_names = ["(none)"] + [p.name for p in profiles]
    chosen_profile = st.selectbox("Apply a saved import profile", profile_names, key="imp_profile",
                                  help="A profile repeats a saved read, mapping and derivation in one click.")

    if st.button("Read data", type="primary", disabled=source is None, key="imp_read"):
        try:
            if chosen_profile != "(none)":
                profile = next(p for p in profiles if p.name == chosen_profile)
                ds = apply_profile(source, profile, name=ds_name)
                st.session_state["imp_ds"] = ds
                st.session_state["imp_mapped"] = ds
                st.session_state["imp_profile_used"] = profile.derivations
            else:
                ds = load(source, file_type="paste" if kind == "Paste a table" else None, sheet=sheet,
                          header_row=int(header_row) - 1, separator=separator, decimal=decimal,
                          dayfirst=dayfirst, name=ds_name, time_unit=time_unit, currency=currency)
                st.session_state["imp_ds"] = ds
                st.session_state["imp_mapped"] = None
                st.session_state.pop("imp_profile_used", None)
            st.session_state["imp_final"] = None
            st.session_state["imp_version"] = st.session_state.get("imp_version", 0) + 1
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the data: {exc}")

    ds = st.session_state.get("imp_ds")
    if ds is None:
        return

    st.success(f"Read **{ds.name}**: {len(ds)} rows × {ds.data.shape[1]} columns.")
    with st.expander("Preview of the original cells"):
        st.dataframe(ds.raw.head(20) if ds.raw is not None else ds.data.head(20), hide_index=True)

    # ---------------------------------------------------------------- 2. Map
    st.subheader("2. Map columns to roles")
    st.caption("Roles are suggested from each column's name and content; change any that are wrong. "
               "Durations need the unit they were recorded in; amounts need to say whether they are "
               "revenue or cost, so profit can be derived.")
    version = st.session_state.get("imp_version", 0)
    table = pd.DataFrame([{
        "column": c,
        "detected": info.kind,
        "example": example_value(ds.data[c]),
        "role": ROLE_LABELS.get(info.role, "(not used)"),
        "duration unit": (info.unit if info.unit in TIME_UNITS else ds.time_unit) if info.role == "duration" else "",
        "amount kind": info.amount_kind or ("other" if info.role == "amount" else ""),
        "why": info.role_reason or "",
    } for c, info in ds.columns.items() if c in ds.data.columns])
    edited = st.data_editor(
        table, hide_index=True, width="stretch", key=f"imp_editor_{version}",
        disabled=["column", "detected", "example", "why"],
        column_config={
            "role": st.column_config.SelectboxColumn("role", options=list(ROLE_LABELS.values()), required=True),
            "duration unit": st.column_config.SelectboxColumn("duration unit", options=["", *TIME_UNITS]),
            "amount kind": st.column_config.SelectboxColumn("amount kind", options=["", *AMOUNT_KINDS]),
        },
    )
    if st.button("Confirm mapping", type="primary", key="imp_confirm"):
        mapping = ColumnMapping()
        for _, row in edited.iterrows():
            role = LABEL_TO_ROLE.get(row["role"])
            mapping.roles[row["column"]] = role
            if role == "duration" and row["duration unit"]:
                mapping.units[row["column"]] = row["duration unit"]
            if role == "amount":
                mapping.amount_kinds[row["column"]] = row["amount kind"] or "other"
        try:
            st.session_state["imp_mapped"] = apply_mapping(ds, mapping)
            st.session_state["imp_final"] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Mapping not applied: {exc}")

    mapped = st.session_state.get("imp_mapped")
    if mapped is None:
        return

    # ---------------------------------------------------------------- 3. Derive
    st.subheader("3. Derive measures")
    avail = available_derivations(mapped)
    used = st.session_state.get("imp_profile_used")
    selected = []
    if avail:
        st.caption(f"Added as columns, in {mapped.time_unit} for times and {mapped.currency} for money.")
        for key, text in avail.items():
            if st.checkbox(text, value=True, key=f"imp_der_{key}_{version}"):
                selected.append(key)
    else:
        st.caption("No row-level measures can be derived from these roles.")

    tables = {}
    has_ts = mapped.first_with_role("timestamp") is not None
    if has_ts:
        tables.update({"count per hour": ("h",), "count per day": ("D",), "count per week": ("W",),
                       "hour-of-day profile": ("profile", "hour"), "day-of-week profile": ("profile", "weekday")})
    if all(mapped.first_with_role(r) for r in ("resource", "start", "end")):
        tables["picks per resource-hour"] = ("resource",)
    chosen_tables, weekdays_only, hours = [], False, None
    if tables:
        chosen_tables = st.multiselect("Period and profile tables (saved as separate datasets)", list(tables),
                                       key=f"imp_tables_{version}")
        if any(tables[t][0] in ("h", "D") for t in chosen_tables):
            cc = st.columns(2)
            weekdays_only = cc[0].checkbox("Weekdays only", value=True, key="imp_weekdays")
            if "count per hour" in chosen_tables:
                hours = cc[1].slider("Operating hours for hourly counts", 0, 24, (8, 18), key="imp_hours")

    if st.button("Add to session", type="primary", key="imp_add"):
        try:
            final = mapped.copy()
            if used is None or selected:
                already = {d.split(":")[0] for d in final.derivations}
                derive(final, [k for k in selected if k not in already])
            S.add_dataset(final)
            added = [final.name]
            for t in chosen_tables:
                spec = tables[t]
                if spec[0] in ("h", "D", "W"):
                    new = period_table(final, spec[0], weekdays_only=weekdays_only,
                                       hours=hours if spec[0] == "h" else None)
                elif spec[0] == "profile":
                    new = profile_table(final, spec[1])
                else:
                    new = picks_per_resource_hour(final)
                S.add_dataset(new)
                added.append(new.name)
            state.set_active(final.name)
            st.session_state["imp_final"] = final.name
            st.success("Added to the session: " + ", ".join(f"**{a}**" for a in added))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not derive the measures: {exc}")

    final_name = st.session_state.get("imp_final")
    if not final_name or final_name not in S.datasets:
        return
    final = S.datasets[final_name]

    # ---------------------------------------------------------------- 4. Check
    st.subheader("4. Data checks and cleaning log")
    for c in quality_checks(final):
        icon, lab = STATUS.get(c.status, ("•", c.status))
        st.markdown(f"{icon} **{lab}**: {c.summary}")
        if c.detail:
            st.caption(c.detail)
    st.dataframe(final.describe_columns(), hide_index=True, width="stretch")
    with st.expander(f"Cleaning log ({len(final.log)} steps)"):
        st.dataframe(final.log.to_frame()[["step", "action", "column", "detail", "rows_affected"]],
                     hide_index=True, width="stretch")
    with st.expander("Clean data (first 50 rows)"):
        st.dataframe(final.data.head(50), hide_index=True)

    st.page_link("pages/analyse.py", label="Go to Analyse", icon=":material/arrow_forward:")

    with st.form("imp_save_profile"):
        st.markdown("**Save as import profile**: next time this export loads in one click.")
        pname = st.text_input("Profile name", value=f"{final.name} export")
        if st.form_submit_button("Save profile"):
            try:
                path = save_profile(profile_from_dataset(final, pname))
                st.success(f"Saved profile '{pname}' to {path}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Profile not saved: {exc}")


page()
