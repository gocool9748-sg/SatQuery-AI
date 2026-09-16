# 🛰️ SatQuery AI

SatQuery AI is an AI-powered satellite image analysis application that allows users to upload satellite/remote-sensing imagery and ask questions in natural language. The project combines AI-based image understanding with GIS/raster analysis and vegetation analysis.

---

## 📌 Project Stack

- **Python** – Core application and AI pipeline
- **Streamlit** – Web interface
- **Google Gemini Vision** – Natural-language satellite image understanding
- **QGIS / GDAL** – Geospatial and raster analysis
- **NDVI** – Vegetation analysis
- **YOLO / XView-YOLO** – Object/building detection
- **Git + GitHub + Git LFS** – Version control and large-file management

---

# 🚀 Complete Setup Guide for a New PC

Follow these steps in order.

## 1. Install the required software

Install:

### Python

Install **Python 3.11 or 3.12**.

During installation on Windows, make sure to check:

```text
Add Python to PATH
```

Verify:

```powershell
python --version
```

You should get something similar to:

```text
Python 3.11.x
```

---

### Git

Install Git from the official Git website.

Verify:

```powershell
git --version
```

---

### Git LFS

This project contains large model/data files, so Git LFS is required.

Install Git LFS and then run:

```powershell
git lfs install
```

Verify:

```powershell
git lfs version
```

---

### VS Code

Install Visual Studio Code.

Recommended VS Code extensions:

- Python
- Pylance
- GitHub Pull Requests and Issues

---

# 2. Clone the Repository

Open VS Code.

Open the terminal:

```text
Ctrl + `
```

Clone the project:

```powershell
git clone https://github.com/gocool9748-sg/SatQuery-AI.git
```

Enter the project folder:

```powershell
cd SatQuery-AI
```

Check that the files are present:

```powershell
dir
```

You should see files/folders such as:

```text
app.py
satellite_detector.py
building_detector.py
models/
runs/
xview-yolov3/
*.tif
*.pt
.gitignore
```

---

# 3. Make Sure Git LFS Files Are Downloaded

Run:

```powershell
git lfs pull
```

Then check:

```powershell
git lfs ls-files
```

If the repository contains LFS-managed files, they should appear in the output.

---

# 4. Create a Python Virtual Environment

Inside the `SatQuery-AI` folder, run:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

at the beginning of the terminal line.

Example:

```text
(.venv) PS C:\...\SatQuery-AI>
```

---

# 5. Upgrade pip

Run:

```powershell
python -m pip install --upgrade pip
```

---

# 6. Install Project Dependencies

If a root-level `requirements.txt` exists, run:

```powershell
pip install -r requirements.txt
```

The XView-YOLO component also contains its own requirements file. If needed, install those dependencies with:

```powershell
pip install -r xview-yolov3\requirements.txt
```

If a dependency gives an installation error, do not randomly change the project code. Ask the team first.

---

# 7. Create the `.env` File

⚠️ **Important:** `.env` is intentionally NOT stored in GitHub because it contains private API credentials.

Create a new file in the project root:

```text
.env
```

The location should be:

```text
SatQuery-AI/
│
├── .env
├── app.py
├── satellite_detector.py
├── building_detector.py
└── ...
```

Add the required API key in the format used by the project.

For example:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Replace:

```text
YOUR_GEMINI_API_KEY
```

with your own Gemini API key.

### 🔐 Never commit `.env`

Do NOT run:

```powershell
git add .env
```

The repository's `.gitignore` is configured to keep `.env` out of GitHub.

---

# 8. Verify Important Project Files

Before running the application, make sure the important model/data files are present.

Typical files include:

```text
models/
runs/
xview-yolov3/
ndvi.tif
vegetation_mask.tif
yolo26n.pt
yolo26n-obb.pt
```

The exact files may change as the project develops.

If a large file is missing, run:

```powershell
git lfs pull
```

---

# 9. Select the Correct Python Interpreter in VS Code

In VS Code:

1. Press `Ctrl + Shift + P`
2. Search for:

```text
Python: Select Interpreter
```

3. Select the interpreter inside:

```text
.venv
```

It should look similar to:

```text
.venv\Scripts\python.exe
```

---

# 10. Run the SatQuery AI Application

Make sure the virtual environment is activated:

```powershell
.venv\Scripts\activate
```

Then run:

```powershell
streamlit run app.py
```

Streamlit should display a local address such as:

```text
Local URL: http://localhost:8501
```

Open that address in your browser.

---

# 🧪 11. Basic Test

After the application opens:

1. Upload a supported satellite image.
2. Wait for image processing.
3. Test the satellite/image analysis.
4. Test the natural-language query feature.
5. Test the detection feature if available.
6. Check that the NDVI/raster analysis works if that module is enabled.

If something fails, check the terminal for the exact error message.

---

# 🔄 Working With the Team

## Before starting work

Always get the latest code:

```powershell
git pull
```

If Git LFS files changed:

```powershell
git lfs pull
```

---

## After making changes

Check what changed:

```powershell
git status
```

Add your changes:

```powershell
git add .
```

Create a commit:

```powershell
git commit -m "Describe your changes"
```

Push:

```powershell
git push
```

---

# 🌿 Recommended Team Workflow

Do not directly work on `main` when multiple people are developing the project.

Create your own branch:

```powershell
git checkout -b feature-your-name
```

Example:

```powershell
git checkout -b feature-building-detection
```

Work normally, then:

```powershell
git add .
git commit -m "Improve building detection"
git push -u origin feature-building-detection
```

Then create a Pull Request on GitHub so the team can review and merge the changes.

---

# ⚠️ Important Rules

### 1. Do not commit API keys

Never upload:

```text
.env
```

or paste API keys into Python files.

---

### 2. Do not delete model files

Some `.pt`, `.onnx`, `.tif`, or other large files may be required by the application.

---

### 3. Do not commit the virtual environment

Do not upload:

```text
.venv/
```

It is machine-specific and is already ignored.

---

### 4. Use Git LFS for large files

If adding a new large model or dataset, check with the team before committing it.

---

### 5. Pull before starting work

Use:

```powershell
git pull
git lfs pull
```

before beginning your work.

---

# 🛠️ Common Problems

## `streamlit is not recognized`

Make sure the virtual environment is activated:

```powershell
.venv\Scripts\activate
```

Then:

```powershell
pip install streamlit
```

and run:

```powershell
python -m streamlit run app.py
```

---

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'xyz'
```

First make sure `.venv` is activated.

Then install the missing dependency:

```powershell
pip install xyz
```

If the project has a requirements file, prefer:

```powershell
pip install -r requirements.txt
```

---

## Gemini/API error

Check that `.env` exists in the project root and contains:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do not upload the `.env` file to GitHub.

---

## Model/file not found

Run:

```powershell
git lfs pull
```

Then verify that the required model/data file exists.

---

## Git says there are conflicts

Do not use `git push --force` unless the team specifically tells you to.

Send the conflict/error message to the team and resolve it together.

---

# 📂 Expected Project Structure

The project may look approximately like:

```text
SatQuery-AI/
│
├── .gitignore
├── .env                    # Local only - NOT on GitHub
├── app.py
├── satellite_detector.py
├── building_detector.py
│
├── models/
├── runs/
├── xview-yolov3/
│
├── ndvi.tif
├── vegetation_mask.tif
├── yolo26n.pt
├── yolo26n-obb.pt
│
└── README.md
```

The structure may change as development continues.

---

# 👥 Team Setup Summary

For a new team member, the complete sequence is:

```powershell
git clone https://github.com/gocool9748-sg/SatQuery-AI.git

cd SatQuery-AI

git lfs install
git lfs pull

python -m venv .venv

.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit run app.py
```

If there is no root `requirements.txt`, install the dependencies from the appropriate project/component requirements file before running the application.

Then create your local `.env` file with your own API key.

---

# 📞 If Setup Fails

When asking the team for help, send:

1. The exact command you ran.
2. The complete error message.
3. A screenshot of the terminal.
4. Your Python version:

```powershell
python --version
```

5. Your Git version:

```powershell
git --version
```

This makes debugging much faster.

---

## 🛰️ SatQuery AI

**AI-powered natural-language interaction with satellite imagery.**
