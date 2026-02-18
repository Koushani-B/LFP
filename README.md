# LFP Analysis Setup Guide for VS Code

## Prerequisites
- Python 3.8 or higher
- VS Code installed
- Your Neuralynx data files (.ncs and .nev files)

## Installation Steps

### 1. Install Python Extension in VS Code
- Open VS Code
- Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
- Search for "Python" by Microsoft
- Click Install

### 2. Set Up Virtual Environment
Open the VS Code terminal (View → Terminal or Ctrl+`) and run:

**Windows:**
```bash
python -m venv lfp_env
lfp_env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv lfp_env
source lfp_env/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical operations)
- pandas (data management)
- matplotlib (plotting)
- scipy (signal processing)

### 4. Configure Your Data Path

Open `koushani_lfp_analysis_vscode.py` and find the Configuration section (around line 625).

Update the `base_path` to point to your data directory:

**Windows:**
```python
base_path = r'C:\Users\YourName\Documents\Data\r6627'
```

**Mac/Linux:**
```python
base_path = '/Users/YourName/Documents/Data/r6627'
```

Your data directory should contain a folder with your session name (e.g., `KB_R6627__S27_trace250_500_2k`) 
which contains:
- .ncs files (LFP data)
- .nev files (event markers)

### 5. Run the Analysis

**Option A - Run entire script:**
- Open `koushani_lfp_analysis_vscode.py` in VS Code
- Press F5 or click the Run button in the top right
- Or right-click in the editor and select "Run Python File in Terminal"

**Option B - Run from terminal:**
```bash
python koushani_lfp_analysis_vscode.py
```

## Expected Output

The script will:
1. Load LFP data from .ncs files
2. Create trial structure from event files
3. Perform spectral power analysis across frequency bands
4. Generate statistical comparisons
5. Display visualizations:
   - Box plots comparing power across regions
   - Time-frequency spectrograms for paired trials
   - Time-frequency spectrograms for CS-alone trials
   - Raw LFP traces

## Troubleshooting

### "Module not found" errors
Make sure your virtual environment is activated (you should see `(lfp_env)` in your terminal prompt)

### Path errors
- Use raw strings (r'path') on Windows to avoid backslash issues
- Make sure the path exists and contains the correct session folder

### Performance is slow
This version uses CPU processing (AMD GPU not supported by CuPy). Processing is slower than GPU but still functional. 
Consider analyzing shorter time windows or fewer trials if needed.

### Memory errors
If you run out of memory:
- Close other applications
- Analyze one region at a time
- Reduce the number of trials

## Note on AMD GPU
CuPy (GPU acceleration library) only supports NVIDIA GPUs with CUDA. Since you have an AMD Radeon GPU, 
this version uses CPU-based scipy for all computations. The analysis will work correctly but will be 
slower than the GPU-accelerated version.

## Contact
If you encounter issues, check that:
1. Virtual environment is activated
2. All packages are installed (`pip list`)
3. Data path is correct
4. Data files are in the expected format (.ncs and .nev from Neuralynx)
