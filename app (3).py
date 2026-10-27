from __future__ import annotations

import io
import math
import re
import time
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
WORLD_BANK_API = "https://api.worldbank.org/v2"
COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current age-specific mortality patterns persist."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by the midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people; this is a rate, not a total birth count."),
    "Net migration": ("SM.POP.NETM", "people, net total", "Estimated net number of migrants during the period. It is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "homicides per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt expressed as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including monetary gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services expressed as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Exponential growth/decay", "time", "years", "revenue", "dollars", "A business indicator changing over time."),
    "Engineering": ("Polynomial", "time", "seconds", "position", "meters", "A modeled engineering quantity over time."),
    "Medical": ("Exponential decay", "time", "hours", "concentration", "mg/L", "A simplified concentration model; not medical advice."),
    "Sports science": ("Polynomial", "time", "seconds", "speed", "m/s", "A sports performance measure over time."),
    "Biology": ("Logistic-like (use rational)", "time", "days", "population", "organisms", "A simplified population model."),
    "Bioengineering": ("Exponential growth/decay", "time", "hours", "signal", "arbitrary units", "A simplified biological engineering signal."),
    "Economic forecasting": ("Polynomial", "time", "years", "index", "index points", "A mathematical trend, not a guaranteed forecast."),
    "Weather": ("Trigonometric", "time", "hours", "temperature", "°C", "A periodic temperature approximation."),
    "Custom": ("Polynomial", "x", "units", "f(x)", "units", "Describe your own model and its assumptions."),
}
SUGGESTED = {
    "Polynomial": "0.2*x**3 - 1.5*x**2 + 2*x + 4",
    "Rational": "(x**2 + 2)/(x - 1)",
    "Exponential growth/decay": "8*exp(0.15*x)",
    "Logarithmic": "3*log(x) + 2",
    "Trigonometric": "4*sin(2*pi*x/12) + 10",
    "Logistic-like (use rational)": "100/(1 + 9*exp(-0.5*x))",
}
WB_HEADERS = {"User-Agent": "Hispanic-Heritage-Data-Lab/2.0 (educational project)"}

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title("📈 Hispanic Heritage Data Lab")
st.caption("Historical country data, polynomial modeling, and interactive function analysis for Algebra 2, Precalculus, and Calculus.")
st.info("Brazil is included as a Latin American comparison country. World Bank indicators can have different reporting schedules, definitions, and gaps; actual coverage is reported rather than assumed.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str | None]:
    url = f"{WORLD_BANK_API}/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, headers=WB_HEADERS, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), "The API returned an empty or unexpected response."
        rows = []
        for item in payload[1]:
            year = item.get("date")
            value = item.get("value")
            try:
                year_int = int(year)
            except (TypeError, ValueError):
                continue
            rows.append({"year": year_int, "value": value})
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), "No observations were returned for this period."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        if df.empty:
            return df, "The API response contained no non-null numeric observations."
        return df, None
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "The World Bank API request timed out. Try again later."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank API request failed: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not read the API response: {exc}"

def validate_observations(df: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    clean = df.copy()
    if not {"year", "value"}.issubset(clean.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Data must contain year and value columns."]
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["year", "value"])
    clean = clean[clean["year"] % 1 == 0]
    clean["year"] = clean["year"].astype(int)
    if clean["year"].duplicated().any():
        notes.append("Duplicate years were removed from source data by keeping the first observation.")
        clean = clean.drop_duplicates("year", keep="first")
    if indicator_name in ("Life expectancy at birth",):
        bad = clean["value"] <= 0
    elif indicator_name in ("Unemployment, national estimate", "Consumer price inflation", "Central government debt"):
        bad = clean["value"] < 0 if indicator_name != "Consumer price inflation" else pd.Series(False, index=clean.index)
    elif indicator_name == "Crude birth rate" or indicator_name == "Intentional homicide rate":
        bad = clean["value"] < 0
    else:
        bad = pd.Series(False, index=clean.index)
    if bad.any():
        notes.append(f"{int(bad.sum())} implausible value(s) were excluded by the basic validation rule for {indicator_name}.")
        clean = clean.loc[~bad]
    return clean.sort_values("year").reset_index(drop=True), notes

def poly_fit(df: pd.DataFrame, degree: int):
    x = df["year"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    base = int(np.min(x))
    t = x - base
    if len(np.unique(x)) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {len(np.unique(x))} are available.")
    coef = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coef)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    return base, coef, pred, r2, mae

def poly_expr(coef, base, variable="x"):
    t = sp.Symbol(variable) - int(base)
    return sp.Add(*[sp.Float(float(c), 12) * t**i for i, c in enumerate(coef)])

def fmt_num(value, sig=7):
    try:
        if not np.isfinite(float(value)):
            return "undefined"
        return f"{float(value):.{sig}g}"
    except (ValueError, TypeError):
        return str(value)

def safe_expr(text_expr: str, variable_name: str = "x"):
    expr_text = text_expr.strip().replace("^", "**")
    if not expr_text:
        raise ValueError("Enter a function before analyzing it.")
    if len(expr_text) > 250:
        raise ValueError("Function text is too long (maximum 250 characters).")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s^]*", expr_text):
        raise ValueError("Use familiar math symbols, numbers, parentheses, and functions such as sin, cos, exp, log, and sqrt.")
    var = sp.Symbol(variable_name)
    allowed = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
               "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "E": sp.E}
    try:
        expr = sp.sympify(expr_text, locals={**allowed, variable_name: var}, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse this function. Check parentheses and operators. Details: {exc}") from exc
    if not expr.free_symbols.issubset({var}):
        raise ValueError("Use only the selected independent variable in your function.")
    return var, expr

def evaluate_expr(expr, var, xs):
    fn = sp.lambdify(var, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        vals = fn(np.asarray(xs, dtype=float))
    vals = np.asarray(vals)
    if vals.ndim == 0:
        vals = np.full(np.asarray(xs).shape, vals.item(), dtype=float)
    return np.asarray(vals, dtype=float)

def symbolic_text(expr):
    return sp.sstr(sp.simplify(expr)).replace("**", "^")

def critical_analysis(expr, var, lo, hi):
    d1 = sp.diff(expr, var)
    d2 = sp.diff(d1, var)
    critical, inflections = [], []
    try:
        roots = sp.solveset(sp.Eq(d1, 0), var, domain=sp.Interval(lo, hi))
        if isinstance(roots, sp.FiniteSet):
            for root in roots:
                if root.is_real is False:
                    continue
                rv = float(sp.N(root))
                if lo <= rv <= hi:
                    left = max(lo, rv - max((hi-lo)*1e-5, 1e-5))
                    right = min(hi, rv + max((hi-lo)*1e-5, 1e-5))
                    try:
                        dl, dr = float(sp.N(d1.subs(var, left))), float(sp.N(d1.subs(var, right)))
                        typ = "local maximum" if dl > 0 and dr < 0 else "local minimum" if dl < 0 and dr > 0 else "stationary point (not classified as an extremum)"
                        critical.append((rv, float(sp.N(expr.subs(var, rv))), typ))
                    except Exception:
                        critical.append((rv, float("nan"), "stationary point (test inconclusive)"))
    except Exception:
        pass
    try:
        roots2 = sp.solveset(sp.Eq(d2, 0), var, domain=sp.Interval(lo, hi))
        if isinstance(roots2, sp.FiniteSet):
            for root in roots2:
                if root.is_real is False:
                    continue
                rv = float(sp.N(root))
                if lo < rv < hi:
                    delta = max((hi-lo)*1e-5, 1e-5)
                    try:
                        dl = float(sp.N(d2.subs(var, max(lo, rv-delta))))
                        dr = float(sp.N(d2.subs(var, min(hi, rv+delta))))
                        if dl * dr < 0:
                            inflections.append((rv, float(sp.N(expr.subs(var, rv)))))
                    except Exception:
                        pass
    except Exception:
        pass
    return d1, d2, critical, inflections

def build_chart(df, fitted, base, coef, prediction_rows, sample_increment, expr=None, var=None, lo=None, hi=None,
                show_function=True, show_d1=False, show_d2=False, d1=None, d2=None, moving_x=None, tangent=None, secant=None):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    if not df.empty:
        ax.scatter(df["year"], df["value"], label="Observed historical data", s=24, zorder=4)
        sample = df.iloc[::sample_increment]
        ax.scatter(sample["year"], sample["value"], label=f"Model sample markers (every {sample_increment} year row)", marker="x", s=38)
    if fitted and not df.empty:
        xs = np.linspace(df["year"].min(), df["year"].max(), 500)
        ys = np.polynomial.polynomial.polyval(xs-base, coef)
        ax.plot(xs, ys, label="Polynomial best-fit curve", linewidth=2)
        ax.axvline(df["year"].max(), linestyle="--", linewidth=1, label="Historical boundary")
        if hi is not None and hi > df["year"].max():
            xx = np.linspace(df["year"].max(), hi, 160)
            ax.plot(xx, np.polynomial.polynomial.polyval(xx-base, coef), linestyle=":", linewidth=2, label="Polynomial extrapolation")
    if expr is not None and var is not None and lo is not None and hi is not None and show_function:
        xx = np.linspace(lo, hi, 800)
        yy = evaluate_expr(expr, var, xx)
        # Break the line around invalid values and large jumps, so rational/log discontinuities aren't connected.
        yy[~np.isfinite(yy)] = np.nan
        if len(yy) > 2:
            med = np.nanmedian(np.abs(np.diff(yy))) if np.isfinite(np.nanmedian(np.abs(np.diff(yy)))) else 0
            if med > 0:
                jumps = np.where(np.abs(np.diff(yy)) > max(20 * med, 1e5))[0]
                yy[jumps + 1] = np.nan
        ax.plot(xx, yy, label="Selected function", linewidth=2)
        if show_d1 and d1 is not None:
            ax.plot(xx, evaluate_expr(d1, var, xx), label="First derivative", linestyle="--")
        if show_d2 and d2 is not None:
            ax.plot(xx, evaluate_expr(d2, var, xx), label="Second derivative", linestyle="-.")
        if moving_x is not None:
            y0 = float(evaluate_expr(expr, var, [moving_x])[0])
            ax.scatter([moving_x], [y0], s=65, label="Moving point", zorder=6)
            if tangent is not None and np.isfinite(tangent):
                xt = np.linspace(max(lo, moving_x-(hi-lo)*0.15), min(hi, moving_x+(hi-lo)*0.15), 30)
                yt = y0 + tangent * (xt-moving_x)
                ax.plot(xt, yt, label="Tangent line", linestyle=":")
        if secant is not None:
            xa, xb, ya, yb = secant
            ax.plot([xa, xb], [ya, yb], label="Secant line", linestyle="--")
    if prediction_rows is not None and not prediction_rows.empty:
        ax.scatter(prediction_rows["year"], prediction_rows["estimate"], marker="D", s=48, label="Calculated estimate", zorder=5)
    ax.set_xlabel("Year / independent variable")
    ax.set_ylabel("Value (see selected units)")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def make_pdf(metadata, equation, metrics, findings, preview, results, chart_bytes):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch,
                            topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
             Paragraph(metadata, styles["Normal"]), Spacer(1, 8),
             Paragraph("Model equation", styles["Heading2"]), Paragraph(equation.replace("&", "&amp;"), styles["Normal"]),
             Paragraph("Metrics", styles["Heading2"]), Paragraph(metrics.replace("&", "&amp;"), styles["Normal"]),
             Paragraph("Findings and limitations", styles["Heading2"]), Paragraph(findings.replace("&", "&amp;"), styles["Normal"]),
             Spacer(1, 8)]
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=9.3*inch, height=4.5*inch))
    for title, frame in [("Data preview", preview), ("Calculated-year results", results)]:
        story.append(Paragraph(title, styles["Heading2"]))
        if frame is None or frame.empty:
            story.append(Paragraph("No rows available.", styles["Normal"]))
        else:
            view = frame.head(15).copy().fillna("")
            data = [list(view.columns)] + view.astype(str).values.tolist()
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                     ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
                                     ("FONTSIZE", (0,0), (-1,-1), 7),
                                     ("VALIGN", (0,0), (-1,-1), "TOP")]))
            story.append(tbl)
    doc.build(story)
    return buf.getvalue()

# Sidebar: all historical data controls
with st.sidebar:
    st.header("Historical data")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=list(COUNTRIES.keys()).index("Mexico"))
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_names = st.multiselect("Comparison countries (same indicator)", [x for x in COUNTRIES if x != country_name], default=["Brazil"])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

now_year = date.today().year
start_year, end_year = now_year - 70, now_year - 1
indicator_code, indicator_units, indicator_definition = INDICATORS[indicator_name]
source_df, api_error = fetch_world_bank(COUNTRIES[country_name], indicator_code, start_year, end_year)
source_df, validation_notes = validate_observations(source_df, indicator_name)
if api_error:
    st.error(api_error)
if validation_notes:
    for note in validation_notes:
        st.warning(note)
if source_df.empty:
    st.warning("No validated observations are available for this selection. Polynomial fitting is disabled until usable source data are returned.")
    working_default = pd.DataFrame({"year": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64")})
else:
    working_default = source_df.copy()

key = f"working::{country_name}::{indicator_code}"
if key not in st.session_state:
    st.session_state[key] = working_default.copy()
working_key = key
st.subheader("Source and working data")
st.write(f"**Source:** World Bank indicator `{indicator_code}` · **Retrieved:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} · **Requested period:** {start_year}–{end_year}.")
st.write(f"**Actual coverage:** {int(source_df.year.min()) if not source_df.empty else 'none'}–{int(source_df.year.max()) if not source_df.empty else 'none'} · **Valid observations:** {len(source_df)} · **Units:** {indicator_units}.")
with st.expander("Definition and source details", expanded=False):
    st.write(indicator_definition)
    st.markdown(f"- Indicator metadata: https://api.worldbank.org/v2/indicator/{indicator_code}?format=json")
    st.markdown(f"- Country data API: https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}")
    st.caption("The World Bank API may return null values and country/indicator series may end before the requested end year.")

col_edit1, col_edit2 = st.columns([1, 5])
with col_edit1:
    if st.button("Reset to source", key="reset_source"):
        st.session_state[working_key] = source_df.copy()
        st.session_state.pop("edit_stamp", None)
        st.rerun()
with col_edit2:
    st.caption("Edit the table below to explore a scenario. Source data remain separate. Edited data are labeled user-modified and are not sent back to the World Bank.")
edited = st.data_editor(st.session_state[working_key], num_rows="dynamic", use_container_width=True, key=f"editor::{key}")
edited = edited.copy()
is_edited = not edited.equals(source_df.reset_index(drop=True))
if is_edited:
    st.warning("USER-MODIFIED SCENARIO — model results use the editable table, not only the original source series.")
# Validation for editable data
edit_errors = []
if not {"year", "value"}.issubset(edited.columns):
    edit_errors.append("Editable data must have exactly the columns year and value.")
else:
    edited["year_num"] = pd.to_numeric(edited["year"], errors="coerce")
    edited["value_num"] = pd.to_numeric(edited["value"], errors="coerce")
    if edited[["year_num", "value_num"]].isna().any().any():
        edit_errors.append("Every row must have a numeric integer year and numeric value.")
    if not edited["year_num"].dropna().map(float.is_integer).all():
        edit_errors.append("Years must be integers.")
    if edited["year_num"].dropna().duplicated().any():
        edit_errors.append("Duplicate years are not allowed in working data.")
    if indicator_name in ("Life expectancy at birth",) and (edited["value_num"].dropna() <= 0).any():
        edit_errors.append("Life expectancy values must be greater than zero.")
    if indicator_name in ("Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate", "Central government debt") and (edited["value_num"].dropna() < 0).any():
        edit_errors.append("This indicator cannot have negative values under the app's basic validation rule.")
    working = edited.dropna(subset=["year_num", "value_num"]).copy()
    working = working.rename(columns={"year_num": "year", "value_num": "value"})[["year", "value"]]
    working["year"] = working["year"].astype(int)
    working = working.sort_values("year").reset_index(drop=True)
if edit_errors:
    for error in edit_errors:
        st.error(error)
    working = pd.DataFrame(columns=["year", "value"])
elif not working.empty:
    st.success(f"Working table validated: {len(working)} observations.")
else:
    st.warning("Working table has no valid rows.")

fit = None
fit_error = None
if len(working) >= degree + 1 and working["year"].nunique() >= degree + 1:
    try:
        fit = poly_fit(working, degree)
    except Exception as exc:
        fit_error = str(exc)
else:
    fit_error = f"Degree {degree} requires at least {degree+1} distinct valid years; working data currently have {working['year'].nunique() if not working.empty else 0}."
if fit_error:
    st.warning(f"Polynomial fit unavailable: {fit_error}")

base = coef = fitted_values = r2 = mae = None
equation = "No polynomial equation available."
if fit is not None:
    base, coef, fitted_values, r2, mae = fit
    expr_poly = poly_expr(coef, base, "year")
    equation = f"ŷ = {symbolic_text(expr_poly)}; t = year − {base} (centered time)"
    st.subheader("Polynomial best-fit model")
    st.latex(sp.latex(expr_poly).replace(r"\operatorname", r"\mathrm"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Degree", degree)
    m2.metric("R² (in-sample)", fmt_num(r2))
    m3.metric("MAE (in-sample)", fmt_num(mae) + f" {indicator_units}")
    m4.metric("Observation count", len(working))
    st.caption(f"Centered base year: {base}. Historical fitted range: {working.year.min()}–{working.year.max()}. In-sample mean: {working.value.mean():.6g} {indicator_units}. Metrics describe fit to the same data used to estimate the coefficients.")
    residuals = working.copy()
    residuals["fitted"] = fitted_values
    residuals["residual"] = residuals["value"] - residuals["fitted"]
    with st.expander("Residuals and chronological holdout check"):
        st.dataframe(residuals.rename(columns={"year":"Year","value":"Observed","fitted":"Fitted","residual":"Residual"}), use_container_width=True)
        if len(working) >= max(2*(degree+1), 12):
            split = max(degree+1, int(len(working)*0.8))
            train = working.iloc[:split]
            test = working.iloc[split:]
            try:
                hb, hc, _, _, _ = poly_fit(train, degree)
                test_pred = np.polynomial.polynomial.polyval(test.year.to_numpy()-hb, hc)
                st.write(f"Chronological holdout MAE: {mean_absolute_error(test.value, test_pred):.6g} {indicator_units}; holdout observations: {len(test)}. This score uses the earliest 80% (approximately) for training and later observations for testing.")
            except Exception as exc:
                st.write(f"Holdout comparison unavailable: {exc}")
        else:
            st.write("Chronological holdout comparison unavailable: more observations are needed to keep at least degree + 1 training years and a meaningful test set.")
    if horizon > 0:
        far_x = working["year"].max() + horizon
        far_y = float(np.polynomial.polynomial.polyval(far_x-base, coef))
        if not np.isfinite(far_y) or abs(far_y) > max(1e8, abs(float(working.value.median()))*1e4 + 1e4):
            st.warning(f"Potentially implausible polynomial extrapolation at year {far_x}: {fmt_num(far_y)}. Polynomial extrapolation can grow rapidly and is not a validated forecast.")
    st.caption("Uncertainty note: residuals and MAE summarize observed in-sample errors; they are not 95% prediction intervals. Extrapolation beyond the observed range is especially uncertain.")

st.subheader("Calculate value(s) from the historical polynomial")
single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(now_year, 1950), 2100), step=1, key="single_year")
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if st.button("Calculate Value", disabled=fit is None, key="calc_single"):
    val = float(np.polynomial.polynomial.polyval(int(single_year)-base, coef))
    st.session_state.single_result = {"year": int(single_year), "estimate": val, "base": base, "degree": degree,
                                      "country": country_name, "indicator": indicator_code, "data_hash": hash(tuple(map(tuple, working.values.tolist())))}
data_hash = hash(tuple(map(tuple, working.values.tolist())))
single_result = st.session_state.single_result
if single_result and (single_result.get("country") != country_name or single_result.get("indicator") != indicator_code or single_result.get("degree") != degree or single_result.get("base") != base or single_result.get("data_hash") != data_hash):
    st.session_state.single_result = None
    single_result = None
    st.info("The previous single-year result was invalidated because its underlying model or data changed.")
if single_result:
    yr = single_result["year"]
    within = working.year.min() <= yr <= working.year.max()
    actual_match = working.loc[working.year == yr, "value"]
    if not actual_match.empty:
        st.write(f"**Year {yr}:** fitted estimate = {single_result['estimate']:.8g} {indicator_units}; actual observed value = {actual_match.iloc[0]:.8g} {indicator_units}. These are different quantities.")
    else:
        st.write(f"**Year {yr}:** estimate = {single_result['estimate']:.8g} {indicator_units}. It is {'within' if within else 'outside'} the fitted historical range ({working.year.min()}–{working.year.max()}); {'interpolation' if within else 'extrapolation'}.")

batch_mode = st.radio("Batch year input", ["Year range", "Custom comma-separated years"], horizontal=True)
batch_text = ""
if batch_mode == "Year range":
    c1, c2, c3 = st.columns(3)
    with c1: range_start = st.number_input("Start year", 1950, 2100, min(2000, 2100), key="range_start")
    with c2: range_end = st.number_input("End year", 1950, 2100, min(now_year, 2100), key="range_end")
    with c3: range_step = st.slider("Range step", 1, 10, 1, key="range_step")
    if range_start <= range_end:
        requested_years = list(range(int(range_start), int(range_end)+1, int(range_step)))
    else:
        requested_years = []
        st.error("Start year must be less than or equal to end year.")
else:
    batch_text = st.text_input("Custom years (comma-separated; 1950–2100)", value="2000, 2010, 2020")
    requested_years = []
    invalid_custom = []
    for token in batch_text.split(","):
        token = token.strip()
        if not token:
            continue
        if not re.fullmatch(r"\d{4}", token):
            invalid_custom.append(token)
        else:
            year_value = int(token)
            if 1950 <= year_value <= 2100:
                requested_years.append(year_value)
            else:
                invalid_custom.append(token)
    if invalid_custom:
        st.error("Invalid year input(s): " + ", ".join(invalid_custom) + ". Enter integer years from 1950 through 2100, separated by commas.")
    requested_years = list(dict.fromkeys(requested_years))
if requested_years:
    st.caption("Valid requested years: " + ", ".join(map(str, requested_years)))
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None
if st.button("Calculate batch values", disabled=fit is None or not requested_years, key="calc_batch"):
    values = np.polynomial.polynomial.polyval(np.asarray(requested_years)-base, coef)
    st.session_state.batch_result = pd.DataFrame({"year": requested_years, "estimate": values})
    st.session_state.batch_signature = (country_name, indicator_code, degree, data_hash)
if st.session_state.batch_result is not None:
    if st.session_state.get("batch_signature") != (country_name, indicator_code, degree, data_hash):
        st.session_state.batch_result = None
        st.info("Previous batch results were invalidated because the model or data changed.")
batch_df = st.session_state.batch_result if st.session_state.batch_result is not None else pd.DataFrame(columns=["year","estimate"])
if not batch_df.empty:
    batch_df = batch_df.copy()
    batch_df["classification"] = np.where(batch_df.year.between(working.year.min(), working.year.max()), "within fitted range", "outside fitted range")
    st.dataframe(batch_df, use_container_width=True)

st.subheader("Country comparison")
comparison_frames = []
if not working.empty:
    main = working.copy()
    main["country"] = country_name
    comparison_frames.append(main)
for comp_name in comparison_names:
    comp_df, comp_err = fetch_world_bank(COUNTRIES[comp_name], indicator_code, start_year, end_year)
    comp_df, _ = validate_observations(comp_df, indicator_name)
    if comp_err:
        st.warning(f"{comp_name}: {comp_err}")
    if not comp_df.empty:
        comp_df["country"] = comp_name
        comparison_frames.append(comp_df)
if comparison_frames:
    compare_df = pd.concat(comparison_frames, ignore_index=True)
    figc, axc = plt.subplots(figsize=(10, 4.2))
    for cname, group in compare_df.groupby("country"):
        axc.plot(group.year, group.value, marker=".", linewidth=1.5, label=cname)
    axc.set_xlabel("Year")
    axc.set_ylabel(f"{indicator_name} ({indicator_units})")
    axc.grid(True, alpha=.25); axc.legend(); figc.tight_layout()
    st.pyplot(figc)
    st.caption("Lines can have different coverage because countries report different numbers of years. Gaps are not filled with invented values.")
    plt.close(figc)
else:
    st.info("No country series is available to compare.")

st.divider()
st.header("Function studio: change context, function, parameters, and simulation")
course = st.selectbox("Course mode", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
context = st.selectbox("Context", list(CONTEXTS.keys()))
suggested_kind, default_xname, default_xunit, default_yname, default_yunit, default_context = CONTEXTS[context]
context_description = st.text_input("Context description", default_context)
vcol1, vcol2, vcol3, vcol4 = st.columns(4)
with vcol1: xname = st.text_input("Independent variable", default_xname)
with vcol2: xunit = st.text_input("Independent units", default_xunit)
with vcol3: yname = st.text_input("Dependent variable", default_yname)
with vcol4: yunit = st.text_input("Dependent units", default_yunit)
function_source = st.radio("Function source", ["Historical polynomial best fit", "Suggested function", "Build from parameters", "Enter my own function"], horizontal=False)
kind_options = list(SUGGESTED.keys())
if function_source == "Historical polynomial best fit":
    if fit is None:
        st.warning("A valid historical polynomial fit is not available. Choose a suggested or custom function.")
        function_text = SUGGESTED["Polynomial"]
    else:
        # The historical model is written with a year variable; use x for the function studio.
        function_text = " + ".join([f"({float(c):.12g})*(x-{base})**{i}" if i else f"({float(c):.12g})" for i,c in enumerate(coef)])
        st.caption(f"Using the selected historical indicator model, centered at base year {base}. For this option, x represents calendar year.")
elif function_source == "Suggested function":
    kind = st.selectbox("Suggested function family", kind_options, index=kind_options.index(suggested_kind) if suggested_kind in kind_options else 0)
    function_text = SUGGESTED[kind]
elif function_source == "Build from parameters":
    family = st.selectbox("Parameter model family", ["Polynomial", "Rational", "Exponential growth/decay", "Logarithmic", "Trigonometric"])
    p1, p2, p3, p4 = st.columns(4)
    with p1: a = st.number_input("Parameter a", value=1.0, key="param_a")
    with p2: b = st.number_input("Parameter b", value=0.5, key="param_b")
    with p3: c = st.number_input("Parameter c", value=0.0, key="param_c")
    with p4: d = st.number_input("Parameter d", value=0.0, key="param_d")
    if family == "Polynomial": function_text = f"{a}*x**3 + {b}*x**2 + {c}*x + {d}"
    elif family == "Rational": function_text = f"({a}*x**2 + {b})/(x - {c})"
    elif family == "Exponential growth/decay": function_text = f"{a}*exp({b}*x) + {c}"
    elif family == "Logarithmic": function_text = f"{a}*log(x) + {b}"
    else: function_text = f"{a}*sin({b}*x) + {c}*cos({d}*x)"
    st.code(function_text)
else:
    function_text = st.text_input("Function f(x) (use ^ or ** for powers)", value="x^3 - 3*x")
    st.caption("Supported notation: +, -, *, /, parentheses, powers, sin, cos, tan, exp, log, sqrt, abs, pi, E. Use x as the independent-variable symbol.")
if function_source != "Enter my own function":
    function_text = st.text_input("Editable function expression", value=function_text, key=f"function_expression::{function_source}::{context}")
analysis_col1, analysis_col2 = st.columns(2)
with analysis_col1:
    default_lo, default_hi = (-5.0, 5.0) if xunit not in ("years",) else (float(max(1950, now_year-20)), float(min(2100, now_year+10)))
    interval_start = st.number_input("Analysis interval start", value=float(default_lo), key=f"interval_start::{context}")
with analysis_col2:
    interval_end = st.number_input("Analysis interval end", value=float(default_hi), key=f"interval_end::{context}")
if interval_end <= interval_start:
    st.error("Analysis interval end must be greater than its start.")
    interval_end = interval_start + 1.0
try:
    var, expr = safe_expr(function_text, "x")
    d1, d2 = sp.diff(expr, var), sp.diff(sp.diff(expr, var), var)
    function_valid = True
except Exception as exc:
    function_valid = False
    st.error(str(exc))
    var, expr, d1, d2 = sp.Symbol("x"), sp.Integer(0), sp.Integer(0), sp.Integer(0)

if function_valid:
    st.markdown("**Function:**")
    st.latex("f(x) = " + sp.latex(expr))
    st.markdown("**First derivative:**")
    st.latex("f'(x) = " + sp.latex(d1))
    st.markdown("**Second derivative:**")
    st.latex("f''(x) = " + sp.latex(d2))
    st.caption("Symbolic derivatives are exact for the entered expression; plotted coordinates, root searches, and many classifications are numerical approximations.")
    if course == "Honors Algebra 2":
        st.write("**Algebra 2 focus:** inspect domain and range on the selected interval, end behavior, graph trends, and average rates of change.")
    elif course == "Precalculus":
        st.write("**Precalculus focus:** degree and leading coefficient affect polynomial end behavior; turning points and concavity can be explored through changes in average rates.")
    else:
        st.write("**Calculus focus:** derivatives, tangent/secant lines, critical points, extrema, concavity, and inflection points are analyzed below.")

    # Interactive point and interval controls
    span = interval_end - interval_start
    cpoint, ca, cb = st.columns(3)
    with cpoint: current_x = st.slider("Current input x", float(interval_start), float(interval_end), float((interval_start+interval_end)/2), key=f"current_x::{context}")
    with ca: avg_a = st.slider("Average-rate interval start a", float(interval_start), float(interval_end), float(interval_start), key=f"avg_a::{context}")
    with cb: avg_b = st.slider("Average-rate interval end b", float(interval_start), float(interval_end), float(interval_end), key=f"avg_b::{context}")
    show_f = st.checkbox("Show function", True)
    show_first = st.checkbox("Show first derivative curve", course == "Calculus")
    show_second = st.checkbox("Show second derivative curve", False)
    y_current = float(evaluate_expr(expr, var, [current_x])[0])
    slope_current = float(evaluate_expr(d1, var, [current_x])[0])
    curvature_current = float(evaluate_expr(d2, var, [current_x])[0])
    secant_data = None
    if avg_b != avg_a:
        ya, yb = evaluate_expr(expr, var, [avg_a, avg_b])
        secant_slope = float((yb-ya)/(avg_b-avg_a))
        if np.isfinite(ya) and np.isfinite(yb):
            secant_data = (avg_a, avg_b, float(ya), float(yb))
        else:
            secant_slope = float("nan")
    else:
        secant_slope = float("nan")
    if context == "Engineering" and yname.lower() == "position":
        interpretation = f"At x={current_x:.4g} {xunit}, position is {y_current:.5g} {yunit}; first derivative is velocity ({slope_current:.5g} {yunit}/{xunit}) and second derivative is acceleration ({curvature_current:.5g} {yunit}/{xunit}²)."
    elif context in ("Business", "Economic forecasting"):
        interpretation = f"At x={current_x:.4g} {xunit}, {yname} is {y_current:.5g} {yunit}. The first derivative ({slope_current:.5g} {yunit}/{xunit}) describes its instantaneous growth/decline rate; the second derivative ({curvature_current:.5g} {yunit}/{xunit}²) describes how that rate changes."
    elif context in ("Medical", "Bioengineering"):
        interpretation = f"At x={current_x:.4g} {xunit}, modeled {yname} is {y_current:.5g} {yunit}; the rate is {slope_current:.5g} {yunit}/{xunit} and the rate's change is {curvature_current:.5g} {yunit}/{xunit}². This is an educational model, not a clinical recommendation."
    else:
        interpretation = f"At x={current_x:.4g} {xunit}, {yname} = {y_current:.5g} {yunit}. The first derivative is {slope_current:.5g} {yunit}/{xunit}; the second derivative is {curvature_current:.5g} {yunit}/{xunit}²."
    st.write(interpretation)
    if np.isfinite(secant_slope):
        st.write(f"Average rate of change on [{avg_a:.4g}, {avg_b:.4g}]: **{secant_slope:.6g} {yunit}/{xunit}**.")
    else:
        st.write("Average rate of change is undefined for this interval (equal endpoints or undefined endpoint values).")
    if np.isfinite(slope_current):
        st.write(f"Instantaneous rate of change at x={current_x:.4g}: **{slope_current:.6g} {yunit}/{xunit}**.")
    else:
        st.write("Instantaneous rate is undefined at the selected point.")
    figf = build_chart(pd.DataFrame(), None, None, None, None, 1, expr, var, interval_start, interval_end,
                       show_function=show_f, show_d1=show_first, show_d2=show_second, d1=d1, d2=d2,
                       moving_x=current_x, tangent=slope_current, secant=secant_data)
    st.pyplot(figf, use_container_width=True)
    # Analysis details
    if course == "Calculus":
        d1s, d2s, crit, infl = critical_analysis(expr, var, interval_start, interval_end)
        st.subheader("Calculus analysis")
        if crit:
            st.write("**Critical/stationary points (numerically classified by sign change):**")
            st.dataframe(pd.DataFrame(crit, columns=["x", "f(x)", "classification"]), use_container_width=True)
        else:
            st.write("No finite stationary points were confirmed in the selected interval. A critical point can also occur where the derivative is undefined.")
        if infl:
            st.write("**Verified candidate inflection points (concavity changes):**")
            st.dataframe(pd.DataFrame(infl, columns=["x", "f(x)"]), use_container_width=True)
        else:
            st.write("No inflection point with a verified concavity change was found among the tested second-derivative zeros.")
        grid = np.linspace(interval_start, interval_end, 1200)
        vals, derivs, seconds = evaluate_expr(expr, var, grid), evaluate_expr(d1, var, grid), evaluate_expr(d2, var, grid)
        finite = np.isfinite(vals)
        if finite.any():
            idxs = np.where(finite)[0]
            imin, imax = idxs[np.argmin(vals[finite])], idxs[np.argmax(vals[finite])]
            st.write(f"**Numerical absolute extrema on the sampled closed interval:** minimum ≈ ({grid[imin]:.6g}, {vals[imin]:.6g}); maximum ≈ ({grid[imax]:.6g}, {vals[imax]:.6g}). These are grid-based approximations, not proofs.")
        # Approximate domain restrictions and discontinuities
        invalid = grid[~(np.isfinite(vals))]
        if len(invalid):
            st.warning(f"The function is undefined at {len(invalid)} sampled points in this interval. The graph breaks at non-finite values; inspect log/square-root restrictions and denominator zeros.")
        else:
            st.write("No non-finite values were detected on the sampled interval; this numerical check does not prove the function is continuous everywhere.")
        st.write(f"**Increasing/decreasing behavior (sampled derivative signs):** {int(np.sum(derivs[:-1]*derivs[1:] < 0))} possible derivative sign-change region(s) detected. Use the derivative graph and critical-point table to inspect intervals.")
        st.write(f"**Concavity (sampled second-derivative signs):** concave up where f″ > 0 and concave down where f″ < 0. Detected sign changes in sampled f″: {int(np.sum(seconds[:-1]*seconds[1:] < 0))}.")
    # Contextual visual simulation
    st.subheader("Visual simulation")
    visual_kind = st.selectbox("Visual style", ["Moving object / projectile", "Tank / concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator", "Walking person with prosthetic leg"])
    play_speed = st.select_slider("Playback speed (manual step)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    play_cols = st.columns(3)
    with play_cols[0]:
        if st.button("▶ Play / advance", key="play_advance"):
            st.session_state[f"current_x::{context}"] = min(interval_end, current_x + span * 0.03 * play_speed)
            st.rerun()
    with play_cols[1]:
        if st.button("⏸ Pause", key="pause_sim"):
            st.session_state[f"paused::{context}"] = True
            st.caption("Paused. Use the point slider or Play / advance to continue.")
    with play_cols[2]:
        if st.button("↺ Reset", key="reset_sim"):
            st.session_state[f"current_x::{context}"] = (interval_start+interval_end)/2
            st.rerun()
    fraction = (current_x-interval_start)/span
    if visual_kind == "Walking person with prosthetic leg":
        st.markdown(f"🚶 **Walking simulation** — normalized gait phase: {fraction:.0%}. The moving marker represents the selected model value; this is a conceptual illustration, not a clinical gait or prosthetic design simulation.")
    elif visual_kind == "Moving object / projectile":
        st.markdown(f"🔵 **Moving object** — normalized horizontal progress: {fraction:.0%}; vertical display value f(x) = {fmt_num(y_current)} {yunit}.")
    elif visual_kind == "Tank / concentration indicator":
        st.markdown(f"🧪 **Tank/concentration** — modeled level/concentration: {fmt_num(y_current)} {yunit}. Values are displayed without clipping; negative or extreme values may be physically implausible.")
    elif visual_kind == "Population / cell colony":
        st.markdown(f"🦠 **Population indicator** — model output: {fmt_num(y_current)} {yunit}. A polynomial may not respect biological bounds.")
    elif visual_kind == "Thermometer / weather":
        st.markdown(f"🌡️ **Weather indicator** — modeled reading: {fmt_num(y_current)} {yunit}. A selected trigonometric model is only a simplified periodic approximation.")
    else:
        st.markdown(f"📊 **Business/economic indicator** — model output: {fmt_num(y_current)} {yunit}; current rate: {fmt_num(slope_current)} {yunit}/{xunit}. A fitted curve is not a guarantee of future performance.")
else:
    st.info("Enter a valid function and interval to enable function analysis.")

# Historical polynomial chart and exports
st.divider()
st.header("Visualize and export current selection")
prediction_df = batch_df[["year","estimate"]].copy() if not batch_df.empty else pd.DataFrame(columns=["year","estimate"])
chart_fig = build_chart(working, fit is not None, base, coef, prediction_df, sample_increment,
                        hi=(working.year.max()+horizon if not working.empty else None))
st.pyplot(chart_fig, use_container_width=True)
chart_buffer = io.BytesIO()
chart_fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
chart_bytes = chart_buffer.getvalue()
plt.close(chart_fig)

if printer_mode:
    st.subheader("Printer-friendly summary")
    st.write(f"{country_name} · {indicator_name} · {indicator_code} · {indicator_units}")
    st.write(f"Coverage: {working.year.min()}–{working.year.max()} ({len(working)} observations)" if not working.empty else "No usable data.")
    st.write(f"Model: {equation}")
    st.write(f"R² (in-sample): {fmt_num(r2)}; MAE (in-sample): {fmt_num(mae)}" if fit is not None else "Model unavailable.")
    st.write("Limitations: missing observations are not imputed; the polynomial is descriptive and extrapolation may be unreliable. MAE is not a prediction interval.")
    st.dataframe(working.head(20), use_container_width=True)

export_working = working.copy()
export_working["status"] = "user-modified scenario" if is_edited else "source-derived working data"
summary = pd.DataFrame([{
    "country": country_name, "country_code": COUNTRIES[country_name], "indicator": indicator_name,
    "indicator_code": indicator_code, "units": indicator_units, "definition": indicator_definition,
    "requested_start_year": start_year, "requested_end_year": end_year,
    "actual_start_year": int(working.year.min()) if not working.empty else None,
    "actual_end_year": int(working.year.max()) if not working.empty else None,
    "observation_count": len(working), "retrieved_at": datetime.now().astimezone().isoformat(),
    "edited_data_status": "user-modified scenario" if is_edited else "not edited",
    "degree": degree if fit is not None else None, "base_year": base, "equation": equation,
    "r_squared_in_sample": r2, "mae_in_sample": mae, "mean": float(working.value.mean()) if not working.empty else None,
}])
csv_bytes = export_working.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="historical_working_data.csv", mime="text/csv")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    export_working.to_excel(writer, sheet_name="Working Data", index=False)
    (batch_df if not batch_df.empty else pd.DataFrame(columns=["year","estimate"])).to_excel(writer, sheet_name="Calculated Years", index=False)
xlsx_bytes = xlsx_buf.getvalue()
st.download_button("Download Excel workbook (.xlsx)", xlsx_bytes, file_name="historical_data_lab.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf_metadata = f"Country: {country_name} ({COUNTRIES[country_name]}) · Indicator: {indicator_name} ({indicator_code}) · Units: {indicator_units} · Retrieved: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} · Data status: {'user-modified scenario' if is_edited else 'source-derived'}"
metrics_text = f"Observations: {len(working)}; historical range: {working.year.min() if not working.empty else 'none'}–{working.year.max() if not working.empty else 'none'}; degree: {degree}; base year: {base}; in-sample R²: {fmt_num(r2)}; in-sample MAE: {fmt_num(mae)} {indicator_units}."
findings_text = "Polynomial fit metrics are in-sample. Residuals/MAE are not 95% prediction intervals. Extrapolation can be implausible; no missing values are fabricated. The indicator definitions and units follow the World Bank API metadata where available. A model is descriptive and does not establish historical causes."
pdf_bytes = make_pdf(pdf_metadata, equation, metrics_text, findings_text, export_working.head(20), batch_df, chart_bytes)
st.download_button("Download PDF report", pdf_bytes, file_name="historical_data_report.pdf", mime="application/pdf")

st.caption("Source: World Bank Open Data API. Educational analysis only; not official forecasts, medical guidance, or causal historical explanations.")
