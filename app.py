"""Streamlit interface for a GUM measurement-uncertainty calculation."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re

import altair as alt
import pandas as pd
import streamlit as st
import sympy

from formula_field import formula_input
from engine import (
    DISTRIBUTIONS,
    Budget,
    FormulaError,
    ParsedModel,
    QuantityInput,
    evaluate,
    numeric_column,
    parse_model,
    parse_number_series,
    range_points,
    read_measurement_table,
    richest_column,
    summarize_type_a,
    sweep,
)

COLUMNS = [
    "variabele",
    "waarde",
    "eenheid",
    "type",
    "verdeling",
    "onzekerheid",
    "k",
    "vrijheidsgraden",
    "opmerking",
]

_NAME = re.compile(r"[^\w\- ]+", re.UNICODE)

EXAMPLE_INPUTS = {
    "U": {
        "waarde": "10",
        "eenheid": "V",
        "type": "Type B",
        "verdeling": "Normaal",
        "onzekerheid": "0,1",
        "k": "1",
        "vrijheidsgraden": "∞",
        "opmerking": "spanning",
    },
    "I": {
        "waarde": "2",
        "eenheid": "A",
        "type": "Type B",
        "verdeling": "Normaal",
        "onzekerheid": "0,5%",
        "k": "1",
        "vrijheidsgraden": "∞",
        "opmerking": "stroom, 0,5 procent van de waarde",
    },
}

SAMPLE_CSV = "meting\n1,98\n2,01\n1,99\n2,02\n2,00\n1,97\n2,03\n"

QUANTITY_WIDTHS = [1.05, 1.35, 0.85, 1.1, 1.2, 1.7, 0.65, 1.35, 1.9]
QUANTITY_HEADERS = [
    "Variabele",
    "Waarde",
    "Eenheid",
    "Type",
    "Verdeling",
    "Onzekerheid of halfbreedte",
    "k",
    "Vrijheidsgraden",
    "Opmerking",
]


def main() -> None:
    st.set_page_config(page_title="Meetonzekerheid", layout="wide")
    init_state()
    st.title("Meetonzekerheid")
    st.write(
        "Voer een meetmodel in. Elke grootheid uit de formule krijgt een regel "
        "voor de waarde, het type, de verdeling en de onzekerheid. "
        "De gecombineerde standaardonzekerheid wordt berekend met de eerstegraads GUM-methode. "
        "Grootheden worden als ongecorreleerd behandeld."
    )
    render_storage()
    render_model()
    model = current_model()
    if model is None:
        return
    rows = render_quantity_table(model)
    render_type_a(rows)
    render_budget(model, rows)
    render_sweep(model, rows)


def init_state() -> None:
    st.session_state.setdefault("formule", "P = U*I")
    st.session_state.setdefault("inputs", {name: dict(values) for name, values in EXAMPLE_INPUTS.items()})
    st.session_state.setdefault("formula_generation", 0)
    st.session_state.setdefault("measurements", {})
    st.session_state.setdefault("csv_generation", {})
    st.session_state.setdefault("result_unit", "W")
    st.session_state.setdefault("confidence", 0.95)


def reset_quantity_table() -> None:
    """Drop the editor overlay so the next table is built from saved inputs."""
    st.session_state["_qty_names"] = None
    st.session_state.pop("quantity_editor", None)


def load_example() -> None:
    st.session_state.formule = "P = U*I"
    st.session_state.inputs = {name: dict(values) for name, values in EXAMPLE_INPUTS.items()}
    st.session_state.measurements = {}
    st.session_state.result_unit = "W"
    st.session_state.formula_generation += 1
    st.session_state.sweep_bounds_for = None
    drop_quantity_widgets()
    clear_type_a_widgets()
    assign_quantity_widgets(st.session_state.inputs)
    reset_quantity_table()


def clear_model() -> None:
    st.session_state.formule = ""
    st.session_state.inputs = {}
    st.session_state.measurements = {}
    st.session_state.result_unit = ""
    st.session_state.formula_generation += 1
    st.session_state.sweep_bounds_for = None
    st.session_state.sweep_from = ""
    st.session_state.sweep_to = ""
    drop_quantity_widgets()
    clear_type_a_widgets()
    reset_quantity_table()


def render_model() -> None:
    st.subheader("Meetmodel")
    with st.container(horizontal=True, gap="small"):
        st.button("Voorbeeld", on_click=load_example)
        st.button("Wissen", on_click=clear_model)
    st.session_state.formule = formula_input(
        st.session_state.formule,
        st.session_state.formula_generation,
    )
    st.caption(
        "Griekse tekens en het wortelteken worden op de cursorpositie ingevoegd. "
        "Een subscript schrijf je als U_1. Vermenigvuldigen kan met * of door symbolen tegen elkaar, machten met ^."
    )

    formula = st.session_state.formule.strip()
    if not formula:
        st.info("Voer een formule in, bijvoorbeeld P = U*I.")
        return
    try:
        model = parse_model(formula)
    except FormulaError as exc:
        st.error(str(exc))
        return
    st.latex(model.latex)
    if model.quantities:
        st.caption("Grootheden in het model: " + ", ".join(f"`{name}`" for name in model.quantities))
    else:
        st.caption("Deze formule heeft geen invoergrootheden.")
    st.text_input("Eenheid van het resultaat", key="result_unit")


def current_model() -> ParsedModel | None:
    formula = st.session_state.formule.strip()
    if not formula:
        return None
    try:
        return parse_model(formula)
    except FormulaError:
        return None


def render_quantity_table(model: ParsedModel) -> list[dict]:
    st.subheader("Grootheden")
    st.caption(
        "Waarde en onzekerheid mogen een getal zijn of een formule van de grootheden, "
        "bijvoorbeeld `2*U` of `0,1%*U + 0,02`. "
        "`0,5%` als onzekerheid betekent 0,5 procent van de nominale waarde van díe grootheid. "
        "Bij een normale verdeling is de ingevulde onzekerheid de uitgebreide onzekerheid en deelt het programma door k. "
        "Bij rechthoekig en driehoekig vul je de halfbreedte a in; de standaardonzekerheid wordt a/√3 of a/√6. "
        "Bij Type A worden waarde, onzekerheid en vrijheidsgraden berekend. Die cellen zijn blauw en niet aan te passen."
    )
    header = st.columns(QUANTITY_WIDTHS, gap="small", vertical_alignment="bottom")
    for column, title in zip(header, QUANTITY_HEADERS):
        column.markdown(f"**{title}**")
    rows = []
    for name in model.quantities:
        ensure_quantity_fields(name)
        rows.append(render_quantity_row(name))
    st.session_state.inputs = {row["variabele"]: _clean_row(row) for row in rows}
    st.session_state.quantity_base = pd.DataFrame(rows, columns=COLUMNS)
    return rows


def render_quantity_row(name: str) -> dict:
    summary = None
    is_type_a = st.session_state[field_key(name, "type")] == "Type A"
    use_mean = bool(st.session_state.get(f"mean_{name}", True))
    if is_type_a:
        summary, _error = type_a_computation(name)
    locked = calculated_fields(summary, use_mean)
    cells = st.columns(QUANTITY_WIDTHS, gap="small", vertical_alignment="center")
    cells[0].markdown(f"<div style='padding:8px 0'>{html.escape(name)}</div>", unsafe_allow_html=True)
    displayed = {"variabele": name}
    fields = [
        ("waarde", "Waarde", "text"),
        ("eenheid", "Eenheid", "text"),
        ("type", "Type", "type"),
        ("verdeling", "Verdeling", "distribution"),
        ("onzekerheid", "Onzekerheid of halfbreedte", "text"),
        ("k", "k", "text"),
        ("vrijheidsgraden", "Vrijheidsgraden", "text"),
        ("opmerking", "Opmerking", "text"),
    ]
    for cell, (column, label, kind) in zip(cells[1:], fields):
        with cell:
            if column in locked and summary is not None:
                value = calculated_text(summary, column)
                calculated_cell(value)
            else:
                if kind == "type":
                    value = st.selectbox(label, ["Type A", "Type B"], key=field_key(name, column), label_visibility="collapsed")
                elif kind == "distribution":
                    value = st.selectbox(label, list(DISTRIBUTIONS), key=field_key(name, column), label_visibility="collapsed")
                else:
                    value = st.text_input(label, key=field_key(name, column), label_visibility="collapsed")
                st.session_state.setdefault("quantity_typed", {}).setdefault(name, {})[column] = (
                    "" if value is None else str(value)
                )
        displayed[column] = "" if value is None else str(value)
    return displayed


def calculated_fields(summary, use_mean: bool) -> set[str]:
    if summary is None:
        return set()
    locked = {"onzekerheid", "vrijheidsgraden"}
    if use_mean:
        locked.add("waarde")
    return locked


def calculated_text(summary, column: str) -> str:
    if column == "waarde":
        return fmt(summary.mean)
    if column == "onzekerheid":
        return fmt(summary.standard_uncertainty)
    return fmt(summary.degrees_of_freedom)


def calculated_cell(text: str) -> None:
    safe = html.escape(text)
    st.html(
        "<div style='background:#e8f1f8;color:#1f4e79;border:1px solid #c5d7e8;"
        "border-radius:8px;padding:7px 12px;min-height:38px;line-height:22px'>"
        f"{safe}</div>"
    )


def render_type_a(rows: list[dict]) -> None:
    selected = [row for row in rows if str(row["type"]).endswith("A")]
    if not selected:
        return
    st.subheader("Type A-meetwaarden")
    st.download_button(
        "Download voorbeeld-CSV",
        SAMPLE_CSV,
        file_name="meetwaarden.csv",
        mime="text/csv",
    )
    st.caption(
        "Typ de meetwaarden, of laad ze uit een CSV. "
        "Een decimale komma mag. Een pooled estimate is optioneel: dan is de standaardonzekerheid s_p/√n."
    )
    for row in selected:
        render_type_a_quantity(str(row["variabele"]))


def render_type_a_quantity(name: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{name}**")
        apply_csv_upload(name)
        st.text_area(
            "Meetwaarden",
            key=f"series_{name}",
            height=140,
            placeholder="1,98\n2,01\n1,99",
            help="Eén getal per regel, of gescheiden door spaties of puntkomma's.",
        )
        pooled_col, dof_col = st.columns(2)
        with pooled_col:
            st.text_input("Pooled estimate s_p", key=f"pooled_{name}", placeholder="optioneel")
        with dof_col:
            st.text_input("Vrijheidsgraden van de pooled estimate", key=f"pooled_dof_{name}")
        st.button("Wis meetwaarden", key=f"clear_series_{name}", on_click=clear_series, args=(name,))
        summary, error = type_a_computation(name)
        if error:
            st.error(error)
        use_mean = st.checkbox("Gebruik het gemiddelde als nominale waarde", value=True, key=f"mean_{name}")
        pooled_active = bool(str(st.session_state.get(f"pooled_{name}", "")).strip())
        if pooled_active:
            st.caption("De standaardonzekerheid is s_p gedeeld door de wortel van het aantal waarnemingen.")
        else:
            st.checkbox(
                "Standaardonzekerheid van het gemiddelde (s/√n). Uitvinken gebruikt s zelf.",
                value=True,
                key=f"sqrtn_{name}",
            )
        stored = st.session_state.measurements.setdefault(name, {})
        stored["use_mean"] = use_mean
        stored["of_mean"] = bool(st.session_state.get(f"sqrtn_{name}", True))
        if summary is None:
            if not error:
                st.info("Voer minstens twee meetwaarden in, of één meetwaarde samen met een pooled estimate.")
            return
        values = stored.get("values") or []
        metrics = st.columns(5)
        metrics[0].metric("n", f"{summary.count}")
        metrics[1].metric("gemiddelde", fmt(summary.mean))
        metrics[2].metric("standaardafwijking s", "—" if summary.stdev is None else fmt(summary.stdev))
        metrics[3].metric("standaardonzekerheid u", fmt(summary.standard_uncertainty))
        metrics[4].metric("vrijheidsgraden", fmt(summary.degrees_of_freedom))
        if values:
            st.line_chart(pd.DataFrame({"meetwaarde": values}), height=180)


def apply_csv_upload(name: str) -> None:
    generation = st.session_state.csv_generation.get(name, 0)
    upload = st.file_uploader(
        "CSV met meetwaarden",
        type=["csv", "txt"],
        key=f"csv_{name}_{generation}",
    )
    stored = st.session_state.measurements.setdefault(name, {})
    if upload is not None:
        token = hashlib.md5(upload.getvalue()).hexdigest()
        if token != stored.get("upload_token"):
            try:
                frame = read_measurement_table(upload.getvalue())
                column = richest_column(frame)
                values = numeric_column(frame, column)
            except FormulaError as exc:
                st.error(str(exc))
            else:
                stored.update(
                    {
                        "data": upload.getvalue(),
                        "filename": upload.name,
                        "column": column,
                        "applied_column": column,
                        "upload_token": token,
                    }
                )
                st.session_state[f"series_{name}"] = "\n".join(fmt(value) for value in values)
                st.rerun()
    if not stored.get("data"):
        return
    try:
        frame = read_measurement_table(stored["data"])
    except FormulaError as exc:
        st.error(str(exc))
        return
    options = [str(column) for column in frame.columns]
    if len(options) < 2:
        return
    default = stored.get("column") if stored.get("column") in options else options[0]
    column = st.selectbox("Kolom", options, index=options.index(default), key=f"col_{name}_{generation}")
    if column != stored.get("applied_column"):
        try:
            values = numeric_column(frame, column)
        except FormulaError as exc:
            st.error(str(exc))
            return
        stored["column"] = column
        stored["applied_column"] = column
        st.session_state[f"series_{name}"] = "\n".join(fmt(value) for value in values)
        st.rerun()


def clear_series(name: str) -> None:
    st.session_state[f"series_{name}"] = ""
    st.session_state[f"pooled_{name}"] = ""
    st.session_state[f"pooled_dof_{name}"] = ""
    stored = st.session_state.measurements.get(name, {})
    for key in ("data", "values", "upload_token", "filename", "column", "applied_column", "error"):
        stored.pop(key, None)
    stored["series"] = ""
    stored["pooled"] = ""
    stored["pooled_dof"] = ""
    stored["pooled_stdev"] = None
    stored["pooled_degrees"] = None
    st.session_state.measurements[name] = stored
    st.session_state.csv_generation[name] = st.session_state.csv_generation.get(name, 0) + 1


def field_key(name: str, column: str) -> str:
    return f"qty_{name}_{column}"


def ensure_quantity_fields(name: str) -> None:
    saved = dict(st.session_state.inputs.get(name, {}))
    defaults = {
        "waarde": saved.get("waarde", ""),
        "eenheid": saved.get("eenheid", ""),
        "type": saved.get("type", "Type B"),
        "verdeling": saved.get("verdeling", "Normaal"),
        "onzekerheid": saved.get("onzekerheid", ""),
        "k": saved.get("k", "1") or "1",
        "vrijheidsgraden": saved.get("vrijheidsgraden", "∞") or "∞",
        "opmerking": saved.get("opmerking", ""),
    }
    # Remember what the user typed. Calculated Type A numbers are shown instead,
    # and must not replace those typed values when the cell becomes editable again.
    typed = st.session_state.setdefault("quantity_typed", {}).setdefault(name, {})
    for column, value in defaults.items():
        text = "" if value is None else str(value)
        typed.setdefault(column, text)
        key = field_key(name, column)
        if key not in st.session_state:
            st.session_state[key] = typed[column]
    if st.session_state[field_key(name, "type")] not in ("Type A", "Type B"):
        st.session_state[field_key(name, "type")] = "Type B"
    if st.session_state[field_key(name, "verdeling")] not in DISTRIBUTIONS:
        st.session_state[field_key(name, "verdeling")] = "Normaal"
    stored = st.session_state.measurements.get(name, {})
    series_key = f"series_{name}"
    if series_key not in st.session_state:
        if stored.get("series"):
            st.session_state[series_key] = str(stored["series"])
        elif stored.get("values"):
            st.session_state[series_key] = "\n".join(fmt(float(value)) for value in stored["values"])
        else:
            st.session_state[series_key] = ""
    st.session_state.setdefault(f"pooled_{name}", str(stored.get("pooled") or ""))
    st.session_state.setdefault(f"pooled_dof_{name}", str(stored.get("pooled_dof") or ""))
    st.session_state.setdefault(f"mean_{name}", bool(stored.get("use_mean", True)))
    st.session_state.setdefault(f"sqrtn_{name}", bool(stored.get("of_mean", True)))


def type_a_computation(name: str):
    stored = st.session_state.measurements.setdefault(name, {})
    try:
        values = parse_number_series(str(st.session_state.get(f"series_{name}", "")))
        pooled_text = str(st.session_state.get(f"pooled_{name}", ""))
        dof_text = str(st.session_state.get(f"pooled_dof_{name}", ""))
        pooled = optional_number(pooled_text, f"de pooled estimate van {name}")
        pooled_dof = optional_number(dof_text, f"de vrijheidsgraden van de pooled estimate van {name}")
        if pooled is None:
            pooled_dof = None
        elif pooled_dof is None:
            raise FormulaError(f"vul de vrijheidsgraden van de pooled estimate van {name} in")
        of_mean = bool(st.session_state.get(f"sqrtn_{name}", True))
        summary = None
        if values:
            summary = summarize_type_a(values, True if pooled is not None else of_mean, pooled, pooled_dof)
    except FormulaError as exc:
        stored["error"] = str(exc)
        stored["values"] = []
        stored["pooled_stdev"] = None
        stored["pooled_degrees"] = None
        return None, str(exc)
    stored["series"] = str(st.session_state.get(f"series_{name}", ""))
    stored["pooled"] = pooled_text.strip()
    stored["pooled_dof"] = dof_text.strip()
    stored["values"] = values
    stored["pooled_stdev"] = pooled
    stored["pooled_degrees"] = pooled_dof
    stored["use_mean"] = bool(st.session_state.get(f"mean_{name}", True))
    stored["of_mean"] = bool(st.session_state.get(f"sqrtn_{name}", True))
    stored["error"] = None
    return summary, None


def optional_number(text: str, label: str) -> float | None:
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise FormulaError(f"{label} is geen getal") from exc
    if not math.isfinite(value):
        raise FormulaError(f"{label} moet een eindig getal zijn")
    return value


def assign_quantity_widgets(inputs: dict) -> None:
    typed_all: dict = {}
    for name, values in inputs.items():
        typed: dict = {}
        for column in COLUMNS:
            if column == "variabele":
                continue
            raw = values.get(column, "")
            text = "" if raw is None else str(raw)
            st.session_state[field_key(name, column)] = text
            typed[column] = text
        typed_all[name] = typed
    st.session_state.quantity_typed = typed_all


def drop_quantity_widgets() -> None:
    st.session_state.pop("quantity_typed", None)
    for key in list(st.session_state.keys()):
        if key.startswith("qty_"):
            st.session_state.pop(key, None)


def clear_type_a_widgets() -> None:
    for key in list(st.session_state.keys()):
        if key == "csv_generation":
            continue
        if key.startswith(("series_", "pooled_", "mean_", "sqrtn_", "csv_", "col_", "clear_series_")):
            st.session_state.pop(key, None)


def render_budget(model: ParsedModel, rows: list[dict]) -> None:
    st.subheader("Onzekerheidsbudget")
    confidence = st.selectbox(
        "Betrouwbaarheidsniveau",
        [0.95, 0.99, 0.9545],
        format_func=lambda value: {0.95: "95%", 0.99: "99%", 0.9545: "95,45% (k = 2 bij oneindig ν)"}[value],
        key="confidence",
    )
    try:
        budget = evaluate(model, [quantity_from_row(row) for row in rows], confidence)
    except FormulaError as exc:
        st.error(str(exc))
        return
    show_budget(budget)


def show_budget(budget: Budget) -> None:
    unit = st.session_state.result_unit.strip()
    suffix = f" {unit}" if unit else ""
    metrics = st.columns(4)
    metrics[0].metric(budget.output, fmt(budget.nominal) + suffix)
    metrics[1].metric("standaardonzekerheid uc", fmt(budget.combined) + suffix)
    metrics[2].metric("uitgebreide onzekerheid U", fmt(budget.expanded) + suffix)
    relative = "—" if budget.relative is None else fmt(100 * budget.relative) + " %"
    metrics[3].metric("relatieve uc", relative)
    st.write(
        f"{budget.output} = {fmt(budget.nominal)} ± {fmt(budget.expanded)}{suffix} "
        f"(k = {fmt(budget.coverage_factor)}, "
        f"{fmt(100 * budget.confidence)} %, "
        f"νeff = {fmt(budget.degrees_of_freedom)})"
    )
    table = pd.DataFrame(
        [
            {
                "grootheid": item.name,
                "waarde": fmt(item.nominal),
                "eenheid": item.unit,
                "type": "A" if item.kind == "A" else "B",
                "verdeling": item.distribution,
                "omrekening": item.conversion,
                "u": fmt(item.standard_uncertainty),
                "ν": fmt(item.degrees_of_freedom),
                "c": fmt(item.sensitivity),
                "c·u": fmt(item.product),
                "aandeel": f"{item.share:.1f} %".replace(".", ","),
                "opmerking": item.comment,
            }
            for item in budget.contributions
        ]
    )
    st.dataframe(table, hide_index=True)
    if budget.contributions:
        with st.expander("Gevoeligheidscoëfficiënten"):
            for item in budget.contributions:
                output = sympy.latex(sympy.Symbol(budget.output))
                quantity = sympy.latex(sympy.Symbol(item.name))
                st.latex(rf"\frac{{\partial {output}}}{{\partial {quantity}}} = {item.sensitivity_latex}")
    with st.expander("Rekenmethode"):
        st.markdown(
            r"""
De standaardonzekerheid van de uitkomst is

$$
u_c = \sqrt{\sum_i c_i^2 u^2(x_i)}, \quad c_i = \frac{\partial f}{\partial x_i}
$$

beoordeeld in de nominale waarden. De effectieve vrijheidsgraden volgen uit Welch–Satterthwaite:

$$
\nu_{\mathrm{eff}} = \frac{u_c^4}{\sum_i (c_i u_i)^4 / \nu_i}
$$

De dekkingsfactor k is het tweezijdige kwantiel van de t-verdeling met die vrijheidsgraden.
Bij oneindig veel vrijheidsgraden is dat het normale kwantiel.
"""
        )


def render_sweep(model: ParsedModel, rows: list[dict]) -> None:
    st.subheader("Verloop")
    if not model.quantities:
        st.caption("Deze formule heeft geen invoervariabele om te variëren.")
        return
    prepare_sweep(model, rows)
    choice, start_box, stop_box = st.columns([2, 1, 1])
    with choice:
        variable = st.selectbox("Invoervariabele", model.quantities, key="sweep_variable")
    with start_box:
        st.text_input("Van", key="sweep_from")
    with stop_box:
        st.text_input("Tot", key="sweep_to")
    st.caption(
        "De overige grootheden blijven op hun ingevulde waarde. "
        "Hangt een waarde of onzekerheid van de gekozen grootheid af, dan wordt die in elk punt opnieuw berekend."
    )
    try:
        quantities = [quantity_from_row(row) for row in rows]
        start = parse_bound(st.session_state.sweep_from, "van")
        stop = parse_bound(st.session_state.sweep_to, "tot")
        step, points = range_points(start, stop)
        curve = sweep(model, quantities, variable, start, stop, st.session_state.confidence)
        table = sweep(model, quantities, variable, start, stop, st.session_state.confidence, points=points)
    except FormulaError as exc:
        st.error(str(exc))
        return
    show_sweep(model, rows, variable, curve, table, step, start, stop)


def prepare_sweep(model: ParsedModel, rows: list[dict]) -> None:
    names = model.quantities
    chosen = st.session_state.get("sweep_variable")
    if chosen not in names:
        chosen = names[0]
        st.session_state.sweep_variable = chosen
        st.session_state.sweep_bounds_for = None
    if st.session_state.get("sweep_bounds_for") == chosen:
        return
    start, stop = default_bounds(nominal_hint(model, rows, chosen))
    st.session_state.sweep_from = start
    st.session_state.sweep_to = stop
    st.session_state.sweep_bounds_for = chosen


def nominal_hint(model: ParsedModel, rows: list[dict], name: str) -> float | None:
    try:
        budget = evaluate(model, [quantity_from_row(row) for row in rows], st.session_state.confidence)
    except FormulaError:
        budget = None
    if budget is not None:
        for item in budget.contributions:
            if item.name == name:
                return item.nominal
    row = next((item for item in rows if str(item["variabele"]) == name), None)
    if row is None:
        return None
    try:
        return float(str(row.get("waarde", "")).strip().replace(",", "."))
    except ValueError:
        return None


def default_bounds(nominal: float | None) -> tuple[str, str]:
    if nominal is None or not math.isfinite(nominal):
        return "0", "1"
    if nominal == 0:
        return "-1", "1"
    if nominal > 0:
        return "0", fmt(2 * nominal)
    return fmt(2 * nominal), "0"


def parse_bound(text: str, label: str) -> float:
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        raise FormulaError(f"vul de grens '{label}' van het bereik in")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise FormulaError(f"de grens '{label}' is geen getal") from exc
    if not math.isfinite(value):
        raise FormulaError(f"de grens '{label}' moet een eindig getal zijn")
    return value


def show_sweep(model: ParsedModel, rows: list[dict], variable: str, curve, table, step: float, start: float, stop: float) -> None:
    unit = unit_of(rows, variable)
    result_unit = st.session_state.result_unit.strip()
    x_title = variable + (f" ({unit})" if unit else "")
    y_title = model.output + (f" ({result_unit})" if result_unit else "")
    u_title = f"U ({result_unit})" if result_unit else "uitgebreide onzekerheid U"
    frame = pd.DataFrame(
        {
            "x": curve.values,
            "resultaat": curve.nominals,
            "onzekerheid": curve.expanded,
        }
    )
    hover = [
        alt.Tooltip("x:Q", title=x_title, format=".6g"),
        alt.Tooltip("resultaat:Q", title=y_title, format=".6g"),
        alt.Tooltip("onzekerheid:Q", title=u_title, format=".6g"),
    ]
    base = alt.Chart(frame).encode(x=alt.X("x:Q", title=x_title))
    result = base.mark_line(color="#1f4e79").encode(
        y=alt.Y("resultaat:Q", title=y_title, axis=alt.Axis(titleColor="#1f4e79")),
        tooltip=hover,
    )
    uncertainty = base.mark_line(color="#b45309").encode(
        y=alt.Y("onzekerheid:Q", title=u_title, axis=alt.Axis(titleColor="#b45309", orient="right")),
        tooltip=hover,
    )
    chart = alt.layer(result, uncertainty).resolve_scale(y="independent").properties(height=380)
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Blauwe lijn: {model.output}. Oranje lijn: uitgebreide onzekerheid bij het gekozen betrouwbaarheidsniveau."
    )
    if curve.skipped:
        st.caption(f"{curve.skipped} punten in het bereik zijn overgeslagen omdat de formule daar niet te berekenen is.")
    share = 100 * step / abs(stop - start)
    st.dataframe(
        pd.DataFrame(
            {
                x_title: [fmt(value) for value in table.values],
                y_title: [fmt(value) for value in table.nominals],
                u_title: [fmt(value) for value in table.expanded],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Stap {fmt(step)} ({fmt(share)}% van het bereik).")
    if table.skipped:
        st.caption(f"{table.skipped} tabelpunten zijn overgeslagen omdat de formule daar niet te berekenen is.")


def unit_of(rows: list[dict], name: str) -> str:
    row = next((item for item in rows if str(item["variabele"]) == name), None)
    if row is None:
        return ""
    return str(row.get("eenheid", "")).strip()


def quantity_from_row(row: dict) -> QuantityInput:
    name = str(row["variabele"])
    kind = "A" if str(row["type"]).endswith("A") else "B"
    stored = st.session_state.measurements.get(name, {})
    if kind == "A" and stored.get("error"):
        raise FormulaError(str(stored["error"]))
    return QuantityInput(
        name=name,
        value_expression=str(row.get("waarde", "")),
        unit=str(row.get("eenheid", "")),
        kind=kind,
        distribution=str(row.get("verdeling", "Normaal")),
        uncertainty_expression=str(row.get("onzekerheid", "")),
        coverage_factor=parse_positive(str(row.get("k", "1")), f"k van {name}", default=1.0),
        degrees_of_freedom=parse_dof(str(row.get("vrijheidsgraden", "∞")), name),
        comment=str(row.get("opmerking", "")),
        measurements=stored.get("values") if kind == "A" else None,
        uncertainty_of_mean=bool(st.session_state.get(f"sqrtn_{name}", stored.get("of_mean", True))),
        use_mean_as_value=bool(st.session_state.get(f"mean_{name}", stored.get("use_mean", True))),
        pooled_stdev=stored.get("pooled_stdev") if kind == "A" else None,
        pooled_degrees_of_freedom=stored.get("pooled_degrees") if kind == "A" else None,
    )


def parse_dof(text: str, name: str) -> float:
    cleaned = text.strip().lower().replace(" ", "")
    if cleaned in {"", "inf", "∞", "oneindig", "onbeperkt"}:
        return math.inf
    try:
        value = float(cleaned.replace(",", "."))
    except ValueError as exc:
        raise FormulaError(f"vrijheidsgraden van {name} zijn ongeldig") from exc
    if value <= 0:
        raise FormulaError(f"vrijheidsgraden van {name} moeten positief zijn")
    return value


def parse_positive(text: str, label: str, default: float) -> float:
    cleaned = text.strip()
    if not cleaned:
        return default
    try:
        value = float(cleaned.replace(",", "."))
    except ValueError as exc:
        raise FormulaError(f"{label} is ongeldig") from exc
    return value


def _clean_row(row: dict) -> dict:
    return {column: "" if row.get(column) is None else str(row.get(column)) for column in COLUMNS if column != "variabele"}


def fmt(value: float) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.6g}".replace(".", ",")


def render_storage() -> None:
    st.subheader("Meting bewaren")
    notice = st.session_state.pop("storage_notice", None)
    if notice:
        st.success(notice)
    st.caption(
        "Download de meting als JSON-bestand en upload dat bestand om hem later weer te openen. "
        "Zo blijft de meting bewaard, ook wanneer de app op een website draait."
    )
    left, right = st.columns(2)
    with left:
        st.text_input("Naam", key="save_name", placeholder="bijvoorbeeld vermogen")
        st.download_button(
            "Download meting",
            data=json.dumps(measurement_payload(), ensure_ascii=False, indent=2),
            file_name=download_filename(st.session_state.get("save_name", "")),
            mime="application/json",
        )
    with right:
        generation = st.session_state.get("measurement_generation", 0)
        upload = st.file_uploader(
            "Open een meting",
            type=["json"],
            key=f"measurement_file_{generation}",
        )
        if upload is not None:
            open_uploaded_measurement(upload)


def measurement_payload() -> dict:
    return {
        "versie": 1,
        "formule": st.session_state.get("formule", ""),
        "eenheid": st.session_state.get("result_unit", ""),
        "betrouwbaarheid": st.session_state.get("confidence", 0.95),
        "grootheden": records_for_save(),
        "type_a": type_a_for_save(),
        "verloop": {
            "variabele": st.session_state.get("sweep_variable") or "",
            "van": st.session_state.get("sweep_from", ""),
            "tot": st.session_state.get("sweep_to", ""),
        },
    }


def open_uploaded_measurement(upload) -> None:
    try:
        payload = json.loads(upload.getvalue().decode("utf-8-sig"))
        apply_payload(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        st.error(f"deze meting kan niet worden geopend: {exc}")
        return
    st.session_state.measurement_generation = st.session_state.get("measurement_generation", 0) + 1
    st.session_state.storage_notice = f"Geopend: {upload.name}."
    st.rerun()


def apply_payload(payload: dict) -> None:
    if payload.get("versie") != 1:
        raise ValueError("onbekende bestandsversie")
    st.session_state.formule = str(payload.get("formule", ""))
    rows = payload.get("grootheden") or []
    st.session_state.inputs = {str(row["variabele"]): _clean_row(row) for row in rows}
    st.session_state.result_unit = str(payload.get("eenheid", ""))
    confidence = payload.get("betrouwbaarheid", 0.95)
    if confidence in (0.95, 0.99, 0.9545):
        st.session_state.confidence = confidence
    drop_quantity_widgets()
    clear_type_a_widgets()
    assign_quantity_widgets(st.session_state.inputs)
    measurements = {}
    for name, item in (payload.get("type_a") or {}).items():
        values = [float(value) for value in item.get("values", [])]
        series = str(item.get("series") or "")
        if not series and values:
            series = "\n".join(fmt(value) for value in values)
        pooled = str(item.get("pooled") or "")
        pooled_dof = str(item.get("pooled_dof") or "")
        measurements[str(name)] = {
            "filename": str(item.get("filename") or ""),
            "column": item.get("column"),
            "values": values,
            "series": series,
            "pooled": pooled,
            "pooled_dof": pooled_dof,
            "use_mean": bool(item.get("use_mean", True)),
            "of_mean": bool(item.get("of_mean", True)),
        }
        st.session_state[f"series_{name}"] = series
        st.session_state[f"pooled_{name}"] = pooled
        st.session_state[f"pooled_dof_{name}"] = pooled_dof
        st.session_state[f"mean_{name}"] = bool(item.get("use_mean", True))
        st.session_state[f"sqrtn_{name}"] = bool(item.get("of_mean", True))
        st.session_state.csv_generation[str(name)] = st.session_state.csv_generation.get(str(name), 0) + 1
    st.session_state.measurements = measurements
    verloop = payload.get("verloop") or {}
    variable = str(verloop.get("variabele") or "")
    st.session_state.sweep_variable = variable or None
    st.session_state.sweep_from = str(verloop.get("van", ""))
    st.session_state.sweep_to = str(verloop.get("tot", ""))
    st.session_state.sweep_bounds_for = variable or None
    st.session_state.formula_generation += 1
    reset_quantity_table()


def records_for_save() -> list[dict]:
    frame = st.session_state.get("quantity_base")
    if isinstance(frame, pd.DataFrame) and len(frame):
        return [
            {"variabele": str(row["variabele"]), **_clean_row(row)}
            for row in frame.to_dict("records")
        ]
    return [
        {"variabele": name, **values}
        for name, values in st.session_state.get("inputs", {}).items()
    ]


def type_a_for_save() -> dict:
    stored_sets = {}
    for name, stored in st.session_state.get("measurements", {}).items():
        series = str(st.session_state.get(f"series_{name}", stored.get("series") or ""))
        pooled = str(st.session_state.get(f"pooled_{name}", stored.get("pooled") or ""))
        pooled_dof = str(st.session_state.get(f"pooled_dof_{name}", stored.get("pooled_dof") or ""))
        values = stored.get("values") or []
        if not values and not series.strip() and not pooled.strip():
            continue
        stored_sets[name] = {
            "filename": stored.get("filename") or "",
            "column": stored.get("column"),
            "values": [float(value) for value in values],
            "series": series,
            "pooled": pooled,
            "pooled_dof": pooled_dof,
            "use_mean": bool(st.session_state.get(f"mean_{name}", stored.get("use_mean", True))),
            "of_mean": bool(st.session_state.get(f"sqrtn_{name}", stored.get("of_mean", True))),
        }
    return stored_sets


def download_filename(name: str) -> str:
    cleaned = _NAME.sub("", str(name)).strip()
    return f"{cleaned or 'meting'}.json"


if __name__ == "__main__":
    main()
