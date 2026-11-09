"""CattleSense AI v2.0: evidence-backed research dashboard (Python 3.11 + Gradio 6)."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import read_first_window, extract_features

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
CLASSES = {'RES': 'Resting while standing', 'RUS': 'Ruminating while standing', 'MOV': 'Moving', 'GRZ': 'Grazing'}
PALETTE = ['#89bdc1', '#a59de2', '#e4b36d']

# Only load models you yourself trained/trust; joblib is pickle-based.
def load_model():
    path = ROOT / 'models' / 'random_forest.joblib'
    if not path.exists():
        return None, 'Trained Phase 1 model missing.'
    try:
        bundle = joblib.load(path)
        if int(bundle['feature_count']) != 17 or int(bundle['window_samples']) != 50:
            return None, 'Model expects a different feature/window configuration.'
        return bundle['model'], None
    except Exception as ex:
        return None, f'Model could not be loaded: {ex}'

MODEL, MODEL_ERROR = load_model()

def read_json(version, filename):
    return json.loads((RES / version / filename).read_text(encoding='utf-8'))

def read_csv(version, filename):
    return pd.read_csv(RES / version / filename)

p1 = read_json('phase1', 'metrics.json')
v11 = read_json('v1_1', 'research_summary.json')
v13 = read_json('v1_3', 'research_summary_v13.json')
summary = read_csv('v1_3', 'model_summary.csv')
by_cow = read_csv('v1_3', 'model_comparison_by_cow.csv')
recall = read_csv('v1_3', 'class_recall_by_cow.csv')
confidence = read_csv('v1_3', 'confidence_threshold_diagnostics.csv')


def pct(value, digits=1):
    return f'{float(value)*100:.{digits}f}%'

best_row = summary.sort_values('mean_accuracy', ascending=False).iloc[0]

CSS = '''
.gradio-container {max-width: 1220px !important; margin: auto !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
#hero {padding: 26px 28px; border: 1px solid var(--border-color-primary); border-radius: 16px; background: linear-gradient(125deg, rgba(34,78,81,.28), rgba(72,66,102,.15));}
#hero h1 {font-size: 2.1rem; letter-spacing: -.035em; margin: 0;}
#hero p {opacity: .85; max-width: 840px;}
.kicker {font-size: .75rem; text-transform: uppercase; letter-spacing: .15em; color:#90bfc2; font-weight:700;}
.stat-card {border:1px solid var(--border-color-primary);padding:18px 19px;border-radius:12px;background:var(--background-fill-secondary);min-height:120px;}
.stat-card .value {font-size:1.95rem;font-weight:760;letter-spacing:-.04em;margin:.25rem 0;}
.stat-card .label {font-size:.88rem;opacity:.75;}
.section-desc {opacity:.8;font-size:.94rem;}
.gr-button-primary {font-weight:700 !important;}
.smallprint {font-size:.84rem;opacity:.75;line-height:1.5;}
'''

def card(label, value, foot):
    return f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div><div class="label">{foot}</div></div>'


def plot_signal(a):
    fig, ax = plt.subplots(figsize=(8.5, 3.2), layout='constrained')
    t = np.arange(a.shape[0]) / 25
    for k, color in enumerate(PALETTE):
        ax.plot(t, a[:, k], color=color, lw=1.9, label='XYZ'[k])
    ax.set_xlabel('Time within selected 50-sample window (s)')
    ax.set_ylabel('Acceleration (input units)')
    ax.grid(alpha=.16)
    ax.legend(ncol=3, loc='upper right')
    return fig


def infer(file, cutoff):
    if MODEL_ERROR:
        raise gr.Error(MODEL_ERROR)
    if file is None:
        raise gr.Error('Pehle 50+ readings wali CSV upload karo.')
    try:
        a = read_first_window(file)
        x = extract_features(a).reshape(1, -1)
        probability = MODEL.predict_proba(x)[0]
        labels = [CLASSES.get(str(c),str(c)) for c in MODEL.classes_]
        probs = {name: float(prob) for name, prob in zip(labels, probability)}
        winning = int(np.argmax(probability))
        top = labels[winning]
        conf = float(probability[winning])
        accept = conf >= cutoff
        status = ('**ACCEPTED** (demo policy)' if accept else '**UNCERTAIN / REJECTED** (demo policy)')
        note = (f'### {status}\n\nTop predicted activity: **{top}**  \n'
                f'Model probability: **{pct(conf)}** · Selected cutoff: **{pct(cutoff)}**\n\n'
                '**This is an experimental Phase 1 Random Forest prediction**, not the v1.3 research model. '
                'A high class probability does not establish clinical or welfare reliability.')
        return probs, note, plot_signal(a)
    except Exception as exc:
        raise gr.Error(f'CSV/model error: {exc}')


def chart_model_summary(metric):
    fig,ax=plt.subplots(figsize=(9,4.1),layout='constrained')
    ordered=summary.sort_values(metric,ascending=True)
    vals=ordered[metric].to_numpy()*100
    ax.barh(ordered['model'], vals, color=PALETTE[:len(ordered)])
    ax.set_xlim(0,100)
    ax.set_xlabel('Six-cow mean (%)' if metric!='mean_rus_recall_supported' else 'Mean on cows with RUS examples (%)')
    for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.2f}%',va='center',fontsize=9)
    ax.set_title({'mean_accuracy':'Accuracy','mean_macro_f1':'Macro F1','mean_rus_recall_supported':'Rumination recall'}[metric])
    ax.grid(axis='x',alpha=.15)
    ax.set_axisbelow(True)
    return fig


def selected_by_cow(model,metric):
    tab=by_cow.loc[by_cow['model']==model].sort_values('test_cow')
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    vals=100*tab[metric].to_numpy()
    ax.bar([f'Cow {int(c)}' for c in tab.test_cow],vals,color='#78acb0')
    ax.set_ylim(0,105)
    ax.set_ylabel('Percent')
    ax.set_title(f'{model} | {"Accuracy" if metric=="test_accuracy" else "Macro F1"}')
    for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.1f}',ha='center',fontsize=9)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    return fig


def show_confidence(cow,model):
    tab=confidence[(confidence.test_cow==int(cow)) & (confidence.model==model)].sort_values('threshold')
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    ax.plot(tab.threshold,100*tab.coverage,marker='o',lw=2,color='#78acb0',label='Coverage')
    ax.plot(tab.threshold,100*tab.accepted_accuracy,marker='s',lw=2,color='#a79cda',label='Accuracy of accepted')
    ax.set_xlim(.48,.97);ax.set_ylim(0,105)
    ax.set_xlabel('Cutoff after validation-fitted temperature scaling')
    ax.set_ylabel('Percent')
    ax.grid(alpha=.17)
    ax.legend()
    ax.set_title(f'Cow {cow} · {model}')
    data=tab[['threshold','coverage','accepted_count','accepted_accuracy']].copy()
    data['coverage']=(100*data['coverage']).round(2)
    data['accepted_accuracy']=(100*data['accepted_accuracy']).round(2)
    return fig,data

models = summary['model'].tolist()
with gr.Blocks(title='CattleSense AI | Research Dashboard v2.0') as app:
    gr.HTML('''<div id="hero"><div class="kicker">Independent research prototype · Version 2.0</div>
    <h1>CattleSense AI</h1><p><b>Evaluating reliable cattle activity recognition from motion data.</b> A transparent demonstration of classification, cross-animal performance and uncertainty. Not a deployed livestock welfare device.</p></div>''')
    with gr.Tab('Overview'):
        gr.Markdown('### Research at a glance')
        gr.Markdown('**Dataset:** Japanese Black Beef Cow Behavior Classification Dataset v2 · six beef cattle · 25 Hz · 50-sample windows · four selected activity labels. Each reported six-cow mean is descriptive, not an external validation estimate.')
        with gr.Row():
            gr.HTML(card('Highest mean accuracy, v1.3',pct(best_row.mean_accuracy,2),'RF + training oversampling · six held-out cows'))
            gr.HTML(card('Highest mean macro F1, v1.3',pct(summary.mean_macro_f1.max(),2),'Six-cow descriptive mean'))
            gr.HTML(card('Ruminating recall, Extra Trees',pct(summary.loc[summary.model=='Extra Trees (27)','mean_rus_recall_supported'].iloc[0],2),'Mean across four supported cows'))
        gr.Markdown('''### The research question
**Can compact behaviour classifiers generalize to unseen cattle while recognizing when a prediction may be unreliable?**

**What is already implemented:** animal-level holdout evaluation, baseline and candidate model comparisons, training-only oversampling experiments, uncertainty diagnostics, and a working local prediction interface.

**What is not yet implemented:** validated dairy-cow performance, on-device TinyML inference, battery measurements, welfare decisions, or a reliable sensor calibration pipeline.''')
        gr.Markdown('### Activity categories')
        gr.Markdown('**RES:** resting while standing · **RUS:** ruminating while standing · **MOV:** moving · **GRZ:** grazing')
        gr.Markdown('**Data source:** [Zenodo record 5849025](https://zenodo.org/records/5849025). The training recordings are not distributed with this app.')
    with gr.Tab('Live model demo'):
        gr.Markdown('### Classify one 2-second movement window')
        gr.Markdown('The deployed model is the **original Phase 1 Random Forest (17 features, four classes)**. Later research models appear as experiment results only; they were not exported for live deployment. CSV requires `x,y,z` or `AccX,AccY,AccZ` and at least 50 recordings at nominal 25 Hz. Upload only files you are permitted to use.')
        with gr.Row():
            csv_input = gr.File(label='Upload your cattle sensor CSV',file_types=['.csv'],type='filepath',scale=2)
            cutoff = gr.Slider(.3,.99,value=.75,step=.01,label='Demo acceptance threshold (not a research-calibrated policy)',scale=1)
        go=gr.Button('Analyze uploaded window',variant='primary')
        with gr.Row():
            scores=gr.Label(label='Model class probabilities')
            message=gr.Markdown('Upload a compatible CSV and click Analyze.')
        wave=gr.Plot(label='First 50 readings')
        go.click(infer,inputs=[csv_input,cutoff],outputs=[scores,message,wave])
        gr.Markdown('**Caution:** CSV without timestamps cannot prove 25 Hz continuity. A model prediction on one uploaded sample is not verified ground truth. Train/test sensor differences and cow-to-cow variation matter.')
    with gr.Tab('Research experiments'):
        gr.Markdown('### Three research iterations · data from your completed Colab experiments')
        with gr.Row():
            gr.HTML(card('v1.1 · Random Forest',pct(v11['mean_accuracy_6_animals'],2),'Six held-out animals'))
            gr.HTML(card('v1.2 · Extra Trees', '82.28%','Expanded 27-feature candidate'))
            gr.HTML(card('v1.3 · Best accuracy',pct(best_row.mean_accuracy,2),'Training-only rebalancing'))
        gr.Dataframe(value=summary.rename(columns={'model':'Model','mean_accuracy':'Mean accuracy','mean_macro_f1':'Mean macro F1','mean_rus_recall_supported':'RUS recall (supported cows)','mean_coverage_selected':'Mean coverage','mean_acc_accepted':'Accepted accuracy'}),interactive=False,label='Experiment v1.3 summary · decimal fractions (0.83 = 83%)')
        metric=gr.Dropdown(choices=[('Mean accuracy','mean_accuracy'),('Mean macro F1','mean_macro_f1'),('Rumination recall','mean_rus_recall_supported')],value='mean_accuracy',label='Compare metric')
        comp=gr.Plot(value=chart_model_summary('mean_accuracy'))
        metric.change(chart_model_summary,inputs=metric,outputs=comp)
        gr.Markdown('### Animal-level generalization')
        with gr.Row():
            choice=gr.Dropdown(choices=models,value='Random Forest (17)',label='Select experiment model')
            cowmetric=gr.Radio(choices=[('Accuracy','test_accuracy'),('Macro F1','test_macro_f1')],value='test_accuracy',label='Animal comparison metric')
        cowplot=gr.Plot(value=selected_by_cow('Random Forest (17)','test_accuracy'))
        choice.change(selected_by_cow,inputs=[choice,cowmetric],outputs=cowplot)
        cowmetric.change(selected_by_cow,inputs=[choice,cowmetric],outputs=cowplot)
        gr.Markdown('### Original six-cow evaluation and experiment figures')
        with gr.Row():
            gr.Image(value=str(RES/'v1_3'/'model_comparison.png'),label='v1.3 model comparison',interactive=False)
            gr.Image(value=str(RES/'v1_3'/'rumination_recall.png'),label='Rumination recall by cow',interactive=False)
        gr.Markdown('**Research caution:** Test folds have already informed successive experiments. Differences are exploratory and should not be presented as results from an untouched external validation set.')
    with gr.Tab('Uncertainty'):
        gr.Markdown('### Coverage vs accuracy of accepted predictions')
        gr.Markdown('Select a held-out animal and one of the evaluated models. These are **offline v1.3 diagnostics with validation-fitted temperature scaling**, not the slider policy used in the live demo. Higher accepted accuracy can come at the cost of rejecting most windows.')
        with gr.Row():
            cow=gr.Dropdown(choices=[str(i) for i in range(1,7)],value='4',label='Test cow')
            confmodel=gr.Dropdown(choices=models,value='Extra Trees (27)',label='Research model')
        cplot=gr.Plot()
        ctable=gr.Dataframe(interactive=False,label='Coverage and accepted accuracy shown as percentages')
        cow.change(show_confidence,inputs=[cow,confmodel],outputs=[cplot,ctable])
        confmodel.change(show_confidence,inputs=[cow,confmodel],outputs=[cplot,ctable])
        app.load(show_confidence,inputs=[cow,confmodel],outputs=[cplot,ctable])
        gr.Markdown('**Important:** These probabilities have not been proven calibrated across farms, sensor types or cattle populations. Results are from only six beef cattle and overlapping windows.')
    with gr.Tab('Method & next steps'):
        gr.Markdown('''### Experimental setup
1. Load the publicly released tri-axial accelerometer recordings (25 Hz).
2. Convert millisecond UNIX timestamps into seconds and reject recording gaps.
3. Extract overlapping 50-sample windows (stride 25) and 17 baseline features or 27 candidate features.
4. Hold one cow out for testing and use a different cow for model selection/threshold tuning. Repeat across six test animals.
5. Compare Random Forest, Extra Trees and training-only oversampling. Evaluate per-cow accuracy, macro F1 and RUS recall where labelled RUS examples exist.
6. Analyze confidence thresholds and the coverage trade-off using validation-based temperature scaling.

### Next research investigations
- More robust rumination recognition (especially failure on Cow 1).
- External dairy-cow datasets and evaluation on data from new sensors/farms.
- Embedded TinyML model conversion, quantization and real RAM/latency/power benchmarks.
- Written animal welfare and ethics review before any animal-mounted tests.

### Known limitations
- **Only six Japanese Black beef cattle.** Not validated on dairy cows.
- Some cows do not have examples of all four classes; no RUS means recall is undefined, **not zero**.
- Window overlap reduces effective statistical independence.
- Later experiments reused evaluation cows across research iterations; there is no untouched external test set.
- There is **no deployed v1.3 model** or embedded sensor benchmark in this package.
- A deployment prototype is **not a welfare or veterinary decision aid**.

### Files and reproducibility
The ZIP includes the actual Phase 1 trained baseline, experiment summary CSVs, original result figures, and prior notebooks. The raw dataset is deliberately not redistributed. The results come from completed Colab runs; training scripts remain separately available in the included notebooks.''')
        gr.Markdown('Dataset: [Zenodo DOI 10.5281/zenodo.5849025](https://doi.org/10.5281/zenodo.5849025) · Original dataset licence: CC BY-NC-ND 4.0. Read original dataset terms before sharing any recordings.')
    gr.HTML('<div class="smallprint">CattleSense AI v2.0 · Independent exploratory research dashboard · Not a validated real-time dairy cow or animal welfare device.</div>')

if __name__ == '__main__':
    app.launch(theme=gr.themes.Soft(), css=CSS)
