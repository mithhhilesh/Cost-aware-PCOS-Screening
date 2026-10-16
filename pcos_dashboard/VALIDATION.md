# Validation and provenance

## What was preserved

All 153 original notebook cells were compared as complete JSON objects, in order, against the supplied notebook. Their sources, execution counts, metadata and saved outputs are unchanged. The original file in Downloads was not modified. New cells add research headings, explanations, comparisons and export support. The revised notebook passes notebook-format validation and every code cell passes Python syntax checks.

## Starter models

The matching `Downloads/PCOS_dataset.csv` contains 541 rows. Reusing the original deterministic preparation steps produces 24 predictors with exactly the original names and order, including leading/trailing spaces.

The saved seven-model winner table in original cell 150 selects Logistic Regression for all primary cost stages under the existing recall > F1 > ROC-AUC > accuracy rule. The starter inference models were fitted on the matching CSV using the original Logistic Regression pipeline and parameters. Their selection metrics come from original cell 108 at the saved four-decimal precision. A2 and A3 share the same configuration. The selected-model export from live notebook execution uses the full-precision seven-model table instead.

## Starter comparison data

The starter dashboard uses the complete saved supplementary single-pass five-fold OOF table from original cell 148: seven models × two feature groups × two analyses = 28 rows. These are the original saved scores, at four-decimal precision. They are not newly calculated 50-fold scores, and no unavailable fold SD is invented. The starter comparison is therefore limited to full and low-cost groups; prediction is available for all four distinct groups.

Running notebook Section 10 exports all original live repeated-CV comparisons for every stage and both analyses. Its dashboard then displays their mean and fold SD. The expensive original cross-validation workflow was not rerun during this revision.

## Checks completed

- Fitted inference and probability-space SHAP worked for all five stage bundles, including the duplicated A2/A3 stage. Duplicate-stage predictions agree.
- On the actual CSV, the held-out Logistic Regression explanation used the first existing outer split, 30 training background rows and 30 validation rows. Backgrounds contained no held-out records. These are illustrative explanations conditional on the prior exploratory candidate selection.
- The complete original RF + LR stacking and soft-voting configurations were separately fitted on the same training fold and explained successfully on a validation record. Their SHAP explanations were not approximated by component feature importances.
- Every SHAP run checked that baseline + all contributions equals the positive-class predicted probability within 0.000001.
- Streamlit's application test harness rendered the app, selected every distinct group, completed the forms, submitted predictions and rendered their explanation tables and plots without application exceptions.
- Empty forms were rejected, values rejected by original cleaning were flagged, and the sensitivity comparison rendered successfully.
- The model-comparison chart, SHAP summary and individual waterfall were visually inspected.

The complete revised notebook was not executed from top to bottom. Original saved results were preserved, the additions were checked individually, and the ready-to-run starter dashboard was exercised with real-data fits. The original Colab data-loading cells remain unchanged; local notebook users must supply the CSV path and skip Drive mounting.

## Clinical scope

This is research screening software. SHAP describes model behavior, not medical causation. The score is not an externally validated or calibrated individual disease risk. Diagnostic overlap, global preprocessing decisions in the original workflow, correlated repeated folds and reuse of CV for selection limit inference. The final notebook section documents these without changing historical results.

References: [SHAP permutation explanation](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Permutation.html), [scikit-learn evaluation pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), and [WHO PCOS](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome).
