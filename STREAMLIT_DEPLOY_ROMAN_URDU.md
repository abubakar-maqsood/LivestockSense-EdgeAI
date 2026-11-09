# LivestockSense EdgeAI — Streamlit Public Deployment (Roman Urdu)

**Ab Gradio ki jagah Streamlit par free public website deploy karni hai.** Original `app.py` safe hai. Nayi entrypoint file `streamlit_app.py` hai.

## Step 1: ZIP download aur extract
- ZIP ko extract karo.
- `streamlit_app.py`, `features.py`, `requirements.txt`, `models/`, `results/` root mein dikhne chahiye.

## Step 2: GitHub mein nayi files upload karo
- Apni existing repository `https://github.com/abubakar-maqsood/LivestockSense-EdgeAI` kholo.
- **Add file → Upload files** dabao.
- Extracted folder ke andar se **`streamlit_app.py` aur `requirements.txt`** choose karo. Baqi existing files repository mein pehle se honi chahiye; unhein dobara upload karne ki zaroorat nahi.
- Github poochhe to `requirements.txt` ko replace/overwrite kar do (is version mein Streamlit dependency added hai).
- Commit message: `Add Streamlit public research dashboard`.
- **Commit changes** click karo.
- Check karo repository root par `streamlit_app.py` visible ho.

## Step 3: Streamlit deployment
- `https://share.streamlit.io/` kholo. GitHub se logged in raho.
- **Create app** dabao.
- **Deploy a public app from GitHub** / **Yup, I have an app** choose karo (wording vary ho sakti hai).
- Repository: `abubakar-maqsood/LivestockSense-EdgeAI`
- Branch: `main`
- Main file path: `streamlit_app.py`
- App URL name: `livestocksense-edgeai` (agar available ho)
- **Deploy** dabao.
- Build logs mein package installation dikh sakti hai. Errors ka screenshot bhejo.

## Step 4: Test website
- Dashboard par Overview / AI prediction / Research experiments / Uncertainty analysis / Methodology tabs verify karo.
- AI prediction mein real permitted sensor CSV upload karo (50+ x,y,z rows). `cattle_test_sample_dataset.csv` sirf demo purpose ke liye ho sakti hai.
- **Important:** Demo Phase 1 model use karta hai. v1.3 metrics separate research experiment hain.
- Apna public `*.streamlit.app` link incognito ya doosre device par open karke check karo.

## Step 5: GitHub About update
- Repository main Code page khol kar right About section ka gear icon click karo.
- Website field mein working `https://...streamlit.app` link paste karo; Save changes.
- README mein public demo URL add karo. GitHub repository aur website dono professor ke saath share kar sakte ho.

## Agar app fail ho
- Streamlit cloud logs ka last error screenshot ya text share karo.
- Check: app entrypoint `streamlit_app.py` hai, `app.py` nahi.
- Check: `models/random_forest.joblib` aur `results/v1_3/model_summary.csv` GitHub mein present hain.
- **Data/license caution:** original raw cattle dataset ya private material public upload na karo.
