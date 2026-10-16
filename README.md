# PCOS research notebook and dashboard

The revised notebook retains all 153 original cells, including their code, saved outputs and execution metadata. Research headings and a final extension add seven-model accuracy charts, automatic per-group model selection, probability-space SHAP, patient explanations and dashboard export.

## Use the notebook

1. Open `PCOS_Screening_Research_SHAP_Streamlit.ipynb` in Jupyter or Colab.
2. Install `requirements.txt` into its Python environment. Restart the kernel after installation.
3. Run the original workflow with your CSV. The existing Colab Drive and CSV-path cells are unchanged. Locally, skip the Drive mount and set the CSV path in the existing data-loading cell.
4. Run Sections 9–10. The notebook writes `pcos_dashboard/`, fits each group's selected model and creates `pcos_dashboard.zip`. The original repeated CV can take substantial time.
5. In Colab, download and extract that ZIP to your computer to launch the dashboard.

## Launch the dashboard locally

From the folder containing `streamlit_app.py` and `artifacts/`, using Python 3.12:

```text
python -m pip install -r requirements-lock.txt
python -m streamlit run streamlit_app.py
```

The supplied `artifacts/` is a locally prepared starter bundle. See `VALIDATION.md` for its comparison provenance and checks. Use its pinned `requirements-lock.txt` for compatibility with its fitted models. Running notebook Section 10 creates a new export directly from all original repeated-CV results and records that provenance in its manifest, with a new lock file for the exporting environment.

Keep the full extracted folder together. The app uses paths relative to its own location, so it can be launched from another working directory. Never load joblib files supplied by untrusted people. Use the same package versions for export and inference; `artifacts/manifest.json` records them. `requirements-lock.txt`, if supplied, records the tested environment.

## Dashboard behavior

- Choose all features, symptoms plus blood tests, symptoms plus vitamin D, or low-cost inputs. A2 and A3 are duplicates and appear once.
- The best candidate follows the notebook's existing screening priority: recall, F1, ROC-AUC, then accuracy. All five standalone models and both hybrids remain eligible.
- Supply every input for that group. Units and observed training ranges are shown; observed ranges are not clinical normal ranges. Binary inputs use No/Yes; cycle regularity uses the dataset codes 2/4. Confirm the original meaning of `Cycle length(days)` before clinical use.
- The app uses the fitted model for that subset, rather than substituting invented values for unavailable groups. Values that the original cleaning marks missing must be corrected before prediction.
- View the model score, screening class at threshold 0.5, SHAP waterfall, signed contribution table and plain-language explanation.
- Compare the seven models by accuracy and other metrics. Orange denotes hybrid models. Error bars are fold standard deviations, not confidence intervals.
- Forms stay in process memory; this code does not save patient inputs or prediction history. There is no authentication or clinical deployment infrastructure.

## Interpretation

SHAP explains the complete predictor's positive-class probability. Positive contributions raise the model score; negative contributions lower it relative to a training reference. They do not identify medical causes, treatment effects or a calibrated individual risk. Correlated features can share attribution.

The notebook's saved primary stage rankings select Logistic Regression under its screening rule. The extension recomputes the selection from live comparison tables rather than hard-coding that name. Clinical diagnosis requires clinician assessment and exclusion of other causes; a negative research screen does not rule out PCOS.

Independent clinical validation, validation of input definitions, fairness assessment and probability calibration are still required. Original analysis limitations are documented in the new final notebook section without altering the original reported results.
