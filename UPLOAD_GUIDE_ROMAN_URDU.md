# CattleSense AI — GitHub aur Hugging Face upload guide

## ZIP extract karna
1. ZIP download karo. Right click > Extract All.
2. Extracted `CattleSense_AI_UPLOAD_READY` folder open karo. Is folder **ke andar** `app.py`, `README.md`, `requirements.txt`, `features.py`, `models`, `results`, `notebooks` hain.
3. `.venv` aur raw data included nahi hain. ZIP ko seedha GitHub par upload karne ke bajaye extract karke andar ki files upload karo.

## GitHub (public source + results)
1. https://github.com/new khol kar repo name `CattleSense-AI` rakho; Public select karo; blank repo create karo.
2. Repository page > **uploading an existing file** (ya Add file > Upload files).
3. Extracted folder **ke andar ki** files and folders ko select/drop karo. Confirm karo files root mein aa rahi hain (README.md root mein).
4. **Commit changes** dabao. Agar browser folder drag/drop accept na kare to GitHub Desktop ya Git CLI se push karo. `.venv` upload mat karo.
5. GitHub page refresh karke `app.py`, `models/`, `results/` aur notebooks dekho.

## Hugging Face (interactive app)
1. https://huggingface.co/join par account banao.
2. https://huggingface.co/new-space par `cattlesense-ai`, Public aur Gradio SDK select karo. Hosting eligibility/charges pehle check karo; paid plan screen aaye to bina consent purchase mat karo.
3. Space > Files > Add file > Upload files mein extracted folder **ke andar ki** files aur nested folders upload karo. `README.md` ka YAML header zaroor rehna chahiye.
4. App tab mein build status check karo. 'Running' hone par demo khol kar CSV prediction aur research tabs test karo. Space start hone mein loading time ho sakta hai.
5. Agar 'Build error' aaye to Logs mein actual error check karo. Hugging Face ke SDK runtime aur local Python versions alag ho sakte hain.

## Important
- Public Github repo aur public Hugging Face Space ke files sab dekh sakte hain; credentials ya private animal data upload mat karo.
- Localhost link ko professor share nahi kar sakte; public Space URL share kar sakte ho.
- Phase 1 deployed model ka Cow 4 test accuracy 63.81% hai. v1.3 83.04% six-cow mean *research result* hai, deployed model ka score nahi.
- Animal welfare ya medical decisions ke liye prototype validate nahi hua hai.
