# Sources and methodology

## Primary historical data source

World Bank Open Data API: https://api.worldbank.org/v2/

The application retrieves JSON from the public World Bank API using the selected country code and indicator code. It requests years from the current calendar year minus 70 through the previous calendar year, inclusive. Actual returned coverage and valid observation count are displayed because series can contain nulls or end early.

## Country codes

- Mexico — MX
- Brazil — BR
- Argentina — AR
- Colombia — CO
- Chile — CL
- Peru — PE
- Venezuela — VE
- Ecuador — EC
- Uruguay — UY

Brazil is included as a Latin American comparison country.

## Indicators

The indicator metadata endpoint is `https://api.worldbank.org/v2/indicator/{INDICATOR_CODE}?format=json`.

| Indicator | Code | Units / interpretation |
|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | Years; expected lifespan under current age-specific mortality patterns |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | Percent of labor force; national estimate |
| GDP per capita | `NY.GDP.PCAP.CD` | Current US dollars per person; GDP divided by midyear population, not average wealth or household income |
| Crude birth rate | `SP.DYN.CBRT.IN` | Annual live births per 1,000 people; a rate, not a birth count |
| Net migration | `SM.POP.NETM` | Net migrant count; not an emigration count |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | Intentional homicides per 100,000 people |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | Annual percentage change in consumer price index |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | Percent of GDP |
| Total reserves including gold | `FI.RES.TOTL.CD` | Current US dollars |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | Percent of GDP |

The application displays links to the indicator metadata and country-series API for the selected series. The API's metadata should be treated as authoritative if a definition or unit changes.

## Mathematical methods

- Polynomial fit: NumPy polynomial least-squares fit using centered time `t = year - base year`.
- In-sample metrics: scikit-learn R² and mean absolute error.
- Holdout: chronological split, training on earlier years and scoring on later years, when the sample is large enough.
- Derivatives: symbolic differentiation with SymPy.
- Plotting: Matplotlib.
- PDF output: ReportLab.
- Excel output: pandas with XlsxWriter.

## Limitations

This is an educational exploratory tool. It does not fabricate missing observations, infer historical causes, or guarantee forecasts. Polynomial extrapolation can be unstable. In-sample errors and residuals are not prediction intervals. Root, extrema, and inflection-point checks use numerical sampling and may not prove all features for every expression. The function studio is not a substitute for validated medical, engineering, or financial models.
