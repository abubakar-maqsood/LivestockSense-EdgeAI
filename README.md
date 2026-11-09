# LivestockSense EdgeAI — CattleSense Research Dashboard v2.0

**Public deployment:** use `streamlit_app.py` with Streamlit Community Cloud. Insert your actual deployed URL here once the app is working.

**Local Streamlit run:**
```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Existing Gradio app:** `app.py` has been preserved; run it with `python app.py` if required.

# LivestockSense EdgeAI — CattleSense Research Dashboard v2.0

A Gradio 6 research portfolio dashboard for an exploratory six-cow accelerometer classification study.

## Scope

- **Live demo**: the real saved Phase 1 Random Forest model (17 features, 50 readings; 25 Hz nominal).
- **Research tabs**: actual figures, metrics and tabulations from v1.1, v1.2 and v1.3 experiments.
- **Not included**: updated deployable v1.3 model, raw dataset, dairy-cow validation or real on-device TinyML inference.

## Windows quick start

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:7860 in your browser. When the previous app is using port 7860, stop it using Ctrl+C first.

## CSV input

Provide a permitted recording with `x,y,z` or `AccX,AccY,AccZ`, 50 or more rows. Only the first 50 samples are evaluated. The app cannot confirm sample timing without timestamp columns. Predictions from uploaded CSVs are not ground-truth observations.

## Research caveats

Animal-level held-out test folds used; only six cattle, some labels absent in specific animals. Repeated exploratory comparisons use the same evaluation subjects across versions; no untouched independent final test. Reported means are descriptive. Confidence rejection requires outside validation.

## Dataset citation

Japanese Black Beef Cow Behavior Classification Dataset (v2): https://zenodo.org/records/5849025 , DOI 10.5281/zenodo.5849025. Licence CC BY-NC-ND 4.0; consult original terms. The ZIP intentionally does not contain the raw recordings.

## Security

`joblib.load` uses Python pickle. Load only trained artifacts from a trusted source. This ZIP uses the previously generated Phase 1 file.

## Notebooks

The `notebooks/` directory contains the prior v1.1–v1.3 source notebooks for reproducing research experiments in Colab with a separately downloaded copy of the original dataset. Do not redistribute source dataset recordings from those notebooks without permission.


## Online demo

Deploy this repository to a Gradio Hugging Face Space to get a public link. The live demo uses the Phase 1 Random Forest model; v1.3 results are report-only.
