"""LivestockSense EdgeAI | Streamlit public research demonstration.

Uses the verified Phase 1 classifier for predictions. Later versions are
research summaries ONLY, not replacement prediction models.
"""
from pathlib import Path
import json
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_features, read_first_window

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
LABELS = {"RES": "Resting while standing", "RUS": "Ruminating while standing", "MOV": "Moving", "GRZ": "Grazing"}

st.set_page_config(page_title="LivestockSense | Edge AI Research", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.23);border-radius:12px;padding:16px;background:rgba(110,132,130,.07)}
h1,h2,h3 {letter-spacing:-.025em}
.small-note {font-size:.88rem;opacity:.8;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_phase1_model():
    # joblib uses pickle; ONLY load the supplied, trusted trained artifact.
    path = ROOT / "models" / "random_forest.joblib"
    if not path.is_file():
        raise FileNotFoundError("models/random_forest.joblib is missing")
    bundle = joblib.load(path)
    if bundle.get("feature_count") != 17 or bundle.get("window_samples") != 50:
        raise ValueError("Trained model does not match 17 features / 50 samples")
    return bundle["model"]

@st.cache_data
def load_data():
    p1 = json.loads((RES / "phase1" / "metrics.json").read_text())
    summary = pd.read_csv(RES / "v1_3" / "model_summary.csv")
    by_cow = pd.read_csv(RES / "v1_3" / "model_comparison_by_cow.csv")
    confidence = pd.read_csv(RES / "v1_3" / "confidence_threshold_diagnostics.csv")
    recall = pd.read_csv(RES / "v1_3" / "class_recall_by_cow.csv")
    v11 = pd.read_csv(RES / "v1_1" / "cross_animal_metrics.csv")
    return p1, summary, by_cow, confidence, recall, v11

try:
    p1, summary, by_cow, confidence, recall, v11 = load_data()
except Exception as exc:
    st.error(f"Research result files couldn't load: {exc}")
    st.stop()

with st.sidebar:
    st.title("LivestockSense")
    st.caption("Edge AI research prototype · v2.0 / Streamlit edition")
    page = st.radio("Explore", ["Overview", "AI prediction", "Research experiments", "Uncertainty analysis", "Methodology & limitations"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Experiment dataset**")
    st.caption("6 Japanese Black beef cattle · tri-axial accelerometer · 25 Hz · 4 selected activity classes")
    st.link_button("Dataset on Zenodo", "https://zenodo.org/records/5849025", use_container_width=True)
    st.caption("Not a clinical, commercial or animal-welfare decision system.")

st.title("LivestockSense EdgeAI")
st.caption("An independent research prototype for behaviour recognition and uncertainty assessment from cattle motion data")

if page == "Overview":
    st.subheader("Project at a glance")
    best = summary.loc[summary.mean_accuracy.idxmax()]
    et = summary.loc[summary.model.str.contains("Extra Trees")].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Best six-cow mean accuracy (v1.3)", f"{best.mean_accuracy:.2%}", help="Descriptive average across 6 test cows; not the deployed model")
    b.metric("Best mean macro F1 (v1.3)", f"{summary.mean_macro_f1.max():.2%}")
    c.metric("Extra Trees mean rumination recall", f"{et.mean_rus_recall_supported:.2%}", help="Average across cows with rumination labels")
    st.info("**Live prediction uses the original Phase 1 Random Forest (17 features).** v1.3 results below refer to separately evaluated research candidates and are not performance claims for the deployed model.")
    st.markdown("### Research question")
    st.write("Can compact models identify cattle activity on unseen animals, and can uncertainty-aware prediction reduce the number of wrong accepted classifications?")
    st.markdown("### Work completed")
    st.write("Public dataset preprocessing; animal-level holdout experiments; Random Forest and Extra Trees comparisons; training-only class oversampling; confidence-versus-coverage analysis; and a working CSV inference demonstration.")
    st.markdown("### Four activity labels")
    st.dataframe(pd.DataFrame([{"Code":k,"Behaviour":v} for k,v in LABELS.items()]), hide_index=True, use_container_width=True)
    st.warning("**Limitations:** Only six Japanese Black beef cattle; dairy-cow generalization, deployed microcontroller inference, energy measurements, and animal-welfare conclusions remain unverified.")

elif page == "AI prediction":
    st.subheader("Classify an accelerometer recording")
    st.write("Upload a CSV with 50 or more motion samples in `x,y,z` or `AccX,AccY,AccZ` columns. The model evaluates only the first 50 readings, nominally two seconds at 25 Hz.")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    cutoff = st.slider("Demo minimum class probability (not a calibrated research cutoff)", .30, .99, .75, .01)
    if st.button("Classify first 50 readings", type="primary"):
        if upload is None:
            st.warning("Pehle CSV file upload karo.")
        else:
            try:
                # Using an in-memory file avoids accepting uploads as filesystem paths.
                arr = read_first_window(io.BytesIO(upload.getvalue()))
                model = load_phase1_model()
                proba = model.predict_proba(extract_features(arr).reshape(1,-1))[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
                name = LABELS.get(str(model.classes_[idx]), str(model.classes_[idx]))
                if conf >= cutoff:
                    st.success(f"Accepted prediction: **{name}** · model class probability **{conf:.1%}**")
                else:
                    st.warning(f"Uncertain / rejected by selected cutoff: top class **{name}** · **{conf:.1%}**")
                st.caption("These are model scores, not independently verified correct behaviour or calibrated confidence.")
                prob_table = pd.DataFrame({"Behaviour": [LABELS.get(str(x), str(x)) for x in model.classes_], "Model probability": proba}).sort_values("Model probability",ascending=False)
                st.bar_chart(prob_table.set_index("Behaviour"))
                wave = pd.DataFrame(arr, columns=["X", "Y", "Z"])
                wave.index = np.arange(len(wave)) / 25.0
                wave.index.name = "Seconds at nominal 25 Hz"
                st.line_chart(wave)
            except Exception as exc:
                st.error(f"Unable to classify uploaded CSV: {exc}")
    st.caption("CSV without timestamps cannot prove sampling continuity. Motion units, sensor placement and sensor orientation must match training conditions. Do not upload private or restricted datasets.")

elif page == "Research experiments":
    st.subheader("Experiment comparison: v1.1 → v1.3")
    st.write("The results below are sourced from saved Colab experiments. Data is shown as descriptive exploratory outcomes, not external independent validation.")
    metrics = summary.copy()
    cols = ["mean_accuracy", "mean_macro_f1", "mean_rus_recall_supported"]
    for col in cols:
        metrics[col] = (100 * metrics[col]).round(2)
    st.dataframe(metrics.rename(columns={"model":"Model", "mean_accuracy":"Mean accuracy (%)", "mean_macro_f1":"Mean macro F1 (%)", "mean_rus_recall_supported":"Mean RUS recall, supported cows (%)"}), hide_index=True, use_container_width=True)
    metric = st.selectbox("Compare models", ["mean_accuracy", "mean_macro_f1", "mean_rus_recall_supported"], format_func=lambda x:{"mean_accuracy":"Accuracy", "mean_macro_f1":"Macro F1", "mean_rus_recall_supported":"Rumination recall"}[x])
    st.bar_chart(summary.set_index("model")[[metric]] * 100, y_label="Percent")
    st.subheader("Performance on different held-out cows")
    model_name = st.selectbox("Model", summary["model"].tolist())
    target = st.radio("Metric", ["test_accuracy", "test_macro_f1"], horizontal=True, format_func=lambda x: "Accuracy" if x=="test_accuracy" else "Macro F1")
    cowtab = by_cow.loc[by_cow.model == model_name].sort_values("test_cow")
    st.bar_chart(pd.DataFrame({"Cow": [f"Cow {int(x)}" for x in cowtab.test_cow], "Percent": cowtab[target].to_numpy() * 100}).set_index("Cow"), y_label="Percent")
    st.subheader("Research figures")
    f1, f2 = st.columns(2)
    f1.image(str(RES / "v1_3" / "model_comparison.png"), caption="v1.3 model comparison", use_container_width=True)
    f2.image(str(RES / "v1_3" / "rumination_recall.png"), caption="Rumination recall per animal", use_container_width=True)
    st.caption("A few class labels were absent in some animals; rumination recall averages only over cows with labeled rumination examples. The same six animals informed iterative model exploration.")

elif page == "Uncertainty analysis":
    st.subheader("Selective prediction: coverage versus accepted accuracy")
    st.write("Offline v1.3 research diagnostics use validation-fitted temperature scaling. These are separate from the uncalibrated demo slider in AI prediction.")
    c1,c2 = st.columns(2)
    cow = c1.selectbox("Held-out cow", sorted(confidence.test_cow.unique().tolist()))
    model_name = c2.selectbox("Experimental model", sorted(confidence.model.unique().tolist()))
    sel = confidence.loc[(confidence.test_cow == cow) & (confidence.model == model_name)].sort_values("threshold").copy()
    if sel.empty:
        st.warning("No diagnostic records for this selection")
    else:
        display = sel[["threshold", "coverage", "accepted_accuracy"]].set_index("threshold") * 100
        display.index.name = "Probability cutoff"
        st.line_chart(display.rename(columns={"coverage":"Coverage (%)", "accepted_accuracy":"Accepted prediction accuracy (%)"}), y_label="Percent")
        shown = sel[["threshold","coverage","accepted_count","accepted_accuracy"]].copy()
        shown["coverage"] = (shown["coverage"]*100).round(2)
        shown["accepted_accuracy"] = (shown["accepted_accuracy"]*100).round(2)
        st.dataframe(shown.rename(columns={"coverage":"Coverage (%)", "accepted_accuracy":"Accepted accuracy (%)"}),hide_index=True,use_container_width=True)
    st.warning("A very high accepted accuracy may be achieved while rejecting most windows. This analysis does not establish calibrated or safe field deployment.")

else:
    st.subheader("Methodology and scope")
    st.markdown("""**Data and preprocessing:** Japanese Black Beef Cow Behavior Classification Dataset v2; X/Y/Z acceleration windows of 50 readings at nominal 25 Hz. Label set: RES, RUS, MOV, GRZ. Dataset timestamps were normalized and checked for discontinuities during training.

**Baseline:** Phase 1 Random Forest trained using selected cattle, tested on held-out Cow 4. Stored model uses 17 statistical features and is the **only** model deployed in this app.

**v1.1:** leave-one-cow-out style evaluation across six cattle.

**v1.2:** compare 17-feature Random Forest against a 27-feature Extra Trees model.

**v1.3:** evaluate class balancing applied to training data only and examine validation-selected confidence thresholds.

**Unresolved:** animal-specific generalization, zero rumination recall on Cow 1 in the experiments, domain shift to dairy cattle, accelerometer orientation, sensor calibration, model compression, low-power firmware, energy and real-world welfare validation.

**Dataset credit:** [Zenodo: Japanese Black Beef Cow Behavior Classification Dataset v2](https://zenodo.org/records/5849025). Original raw recordings are **not** included in this repository. Check the dataset's license before redistribution.

**Research ethics:** This is a software-only experimental prototype; animal-mounted testing should not begin without appropriate ethics/welfare clearance and owner consent.
""")
    st.subheader("Reproducibility")
    st.write("Research notebooks are included in the `notebooks/` folder; saved plots and tabulated results are included under `results/`. Full retraining requires accessing the original dataset separately under its license.")
