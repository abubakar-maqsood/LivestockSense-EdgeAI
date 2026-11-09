# CattleSense AI v2.0 — Roman Urdu Guide

## Purani website safe rakho

`D:\Research Project ABK\CattleSense_AI_Connected_Website` ko delete/overwrite **na** karna. Naye ZIP ko **alag** folder mein extract karo, e.g. `D:\Research Project ABK\CattleSense_AI_Dashboard_v2`.

## Step 1: ZIP download aur extract

ZIP download → right click → Extract All. Extracted folder mein `app.py`, `features.py`, `requirements.txt`, `models`, `results` nazar aane chahiye.

## Step 2: Purani app CMD band karo

Agar pehle wali local site `127.0.0.1:7860` pe chal rahi hai, us CMD window mein `Ctrl+C` dabao. Agar pehle wali site abhi band nahi karna chahte to port ko 7861 kar sakte ho (neeche note).

## Step 3: New folder mein CMD

Windows File Explorer se extracted folder open karo. Upar address bar mein `cmd` likho aur Enter dabao.

## Step 4: Virtual environment banao

```cmd
py -3.11 -m venv .venv
```

## Step 5: Libraries install karo

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Step 6: Website chalao

```cmd
.venv\Scripts\python.exe app.py
```

Chrome mein `http://127.0.0.1:7860` open karo. CMD window band mat karna. Agar 7860 busy ho aur purani site band nahi karni, `app.py` ki last line `app.launch()` ko `app.launch(server_port=7861)` bana ke run karo aur `http://127.0.0.1:7861` kholo.

## Step 7: Tabs ko test karo

1. Overview: research goals aur actual mean results dekho.
2. Live model demo: apni existing 50+ sensor readings wali CSV upload karo; Analyze button dabao. Ye **Phase 1** Random Forest model hai, v1.3 deploy nahi hua.
3. Research experiments: v1.1, v1.2, v1.3 charts aur different cows check karo.
4. Uncertainty: Cow aur model select karke coverage vs accepted accuracy inspect karo.
5. Method & next steps: limitations aur planned embedded research parho.

## Step 8: Github / professor demo

Pehle locally test karo; screenshot aur chhoti screen recording bana lo. Raw dataset, private files, `.venv`, ya untrusted joblib files upload mat karo. Publish karne se pehle source dataset ki licence zaroor check karo.

## Agar error aaye

CMD mein last red traceback ka screenshot bhej dena. Working original site ya original Colab files delete mat karna.
