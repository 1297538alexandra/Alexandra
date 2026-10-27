# Hispanic Heritage Data Lab

A self-contained Streamlit app for exploring World Bank historical indicators and learning polynomial modeling and calculus.

## Files

- `app.py` — complete Streamlit application.
- `requirements.txt` — minimal, unpinned runtime dependencies.
- `sources.md` — data sources, indicator definitions, and methodological notes.

## Run locally

Use Python 3.10 or newer. From the folder containing the files:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

The application calls the public World Bank API at runtime. It does not require API keys or credentials.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy. If startup fails, inspect the app logs and verify the repository file names and Python dependency installation.

## Deploy to Render

Create a **Python Web Service** connected to the GitHub repository.

- Build command:

```bash
pip install -r requirements.txt
```

- Start command (single line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Set no secrets unless you add a separate feature that truly requires them. Never commit credentials to GitHub.

## Verification checklist

- **Source accuracy:** compare the displayed indicator code and definition with the World Bank indicator metadata link in the app.
- **Editing/reset:** edit a year/value cell; confirm the “USER-MODIFIED SCENARIO” label appears; click **Reset to source** and verify source values return.
- **Insufficient years:** choose a higher polynomial degree or delete rows until fewer than degree + 1 distinct years remain; fitting should be disabled with a message.
- **Duplicate years:** add a duplicate year; the working table should show a validation error and no fit should be used.
- **Equation and metrics:** verify the centered-time equation, base year, degree, in-sample R²/MAE, residuals, mean, and observation count.
- **Single/batch agreement:** calculate a year in both tools and compare estimates. Both use the same fitted coefficients and centered base year.
- **Invalid inputs/API failure:** try malformed custom years and an unavailable network connection; expect a readable validation/API message rather than fabricated values.
- **Controls:** change country, indicator, degree, graph sample increment, forecast horizon, comparison countries, printer mode, course, context, function, parameters, variable names/units, interval, point, and average-rate endpoints; confirm the associated display changes.
- **Exports:** open the CSV in a text editor, the XLSX workbook in a spreadsheet program, and the PDF in a PDF reader. Confirm summary, working data, calculated years, metadata, chart, and limitations are present.
- **Deployment:** verify the deployed URL loads, API calls work, edits can be made, and all three exports download.

## Common errors

- **API timeout / HTTP error:** check network access and the World Bank API status; retry later. The app caches successful and unsuccessful responses for up to an hour.
- **No data / missing years:** some indicators and countries have gaps or shorter reporting coverage. The app does not fill gaps with invented values.
- **Polynomial fit unavailable:** degree `d` needs at least `d + 1` distinct valid years.
- **Function parse error:** use `x`, `+ - * /`, parentheses, powers (`^` or `**`), and supported functions `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pi`, `E`.
- **Render/Streamlit startup failure:** inspect deployment logs for missing modules, Python version incompatibility, or a wrong entry-point path. Confirm `requirements.txt` is in the repository root.
- **PDF/XLSX export failure:** inspect logs for ReportLab/XlsxWriter import errors and redeploy after correcting dependencies.

## How the app works

1. A cached HTTP request retrieves the requested World Bank indicator for years `current year - 70` through `current year - 1`.
2. Source observations are validated and kept separate from the editable working table.
3. A user may edit the working table. Reset restores the source-derived table.
4. The polynomial model uses centered time `t = year - base year` to reduce numerical conditioning problems. Degree `d` requires `d + 1` distinct years.
5. R² and MAE are in-sample fit metrics. A chronological holdout MAE is shown only when enough data exist.
6. Single-year and batch estimates use the same polynomial coefficients and centered base year.
7. The function studio uses SymPy for symbolic derivatives and NumPy for numerical plotting. Numerical root classification is approximate and can miss complicated roots or discontinuities.
8. CSV, Excel, and PDF outputs are generated from the current working selection and calculated-year results.

## Mathematical assumptions and limitations

- Polynomial regression describes the selected observations; it does not prove causes or guarantee future behavior.
- Missing World Bank observations are not imputed.
- A polynomial can behave implausibly outside its observed range. Extrapolated estimates are flagged when extreme, but not clipped.
- MAE and residuals are not 95% prediction intervals.
- Symbolic derivatives are exact for the parsed expression, while plotted values, sampled extrema, and sign-change checks are numerical approximations.
- The graph breaks at non-finite function values, but numerical sampling cannot fully characterize every discontinuity.
- Contextual visuals are conceptual displays of model values, not validated medical, engineering, weather, or sports simulations.
