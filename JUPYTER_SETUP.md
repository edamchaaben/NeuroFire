# 🚀 JUPYTER SETUP GUIDE - ANACONDA PYTHON

## ⚡ Quick Fix (Choose One Method)

### **Method 1: Windows Batch File (Easiest)**
```
📁 File: LAUNCH_JUPYTER.bat
👉 Action: Double-click this file
⏱️ Time: 5-10 minutes (installs Jupyter, then opens notebook)
```

This will:
1. ✅ Verify Anaconda Python
2. ✅ Install Jupyter (if needed)
3. ✅ Open your notebook automatically
4. ✅ Display URL: http://localhost:8888

### **Method 2: PowerShell Script**
```powershell
# Right-click PowerShell, "Run as Administrator"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
C:\Users\Edam\Downloads\RL\NeuroFire\LAUNCH_JUPYTER.ps1
```

### **Method 3: Manual Terminal Command**
```powershell
# Open PowerShell in the NeuroFire folder, then:
C:\ProgramData\anaconda3.1\python.exe -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb
```

### **Method 4: Install Jupyter First, Then Launch**
```powershell
# Step 1: Install Jupyter
C:\ProgramData\anaconda3.1\python.exe -m pip install jupyter notebook

# Step 2: Launch notebook
C:\ProgramData\anaconda3.1\python.exe -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb
```

---

## 🔧 Why This Is Happening

The error message indicates:
```
Jupyter command `jupyter-notebook` not found
```

This means:
- ❌ Jupyter is not installed in your Python environment
- ❌ OR Jupyter is installed but the `jupyter-notebook` command is not accessible
- ❌ OR PATH is not properly configured

---

## ✅ Complete Solution

### **Step 1: Verify Anaconda Python**
```powershell
C:\ProgramData\anaconda3.1\python.exe --version
# Should show: Python 3.x.x
```

### **Step 2: Install Jupyter**
```powershell
C:\ProgramData\anaconda3.1\python.exe -m pip install jupyter notebook ipython
```

### **Step 3: Verify Installation**
```powershell
C:\ProgramData\anaconda3.1\python.exe -m jupyter --version
# Should show version information
```

### **Step 4: Launch Notebook**
```powershell
cd "C:\Users\Edam\Downloads\RL\NeuroFire"
C:\ProgramData\anaconda3.1\python.exe -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb
```

---

## 📁 Which File Should I Use?

| Situation | Use This File |
|-----------|---------------|
| **Just want to run the comparison** | `RUN_WITH_ANACONDA.bat` |
| **Want Jupyter notebook** | `LAUNCH_JUPYTER.bat` |
| **Comfortable with PowerShell** | `LAUNCH_JUPYTER.ps1` |
| **Want manual control** | Terminal commands (below) |

---

## 🎯 Fastest Path (No Jupyter Needed)

If Jupyter is problematic, just run the direct comparison:

```batch
RUN_WITH_ANACONDA.bat
```

This will:
- ✅ Use Anaconda Python
- ✅ Install dependencies
- ✅ Run full comparison (3-5 minutes)
- ✅ Generate visualizations
- ✅ No Jupyter needed!

**Result in 3-5 minutes:**
- `neurofire_rl_comparison.png` (8-panel dashboard)
- Console metrics and recommendations

---

## ⚠️ Troubleshooting

### **Problem: "Python not found at C:\ProgramData\anaconda3.1"**

**Solution 1: Find your Anaconda installation**
```powershell
Get-ChildItem "C:\ProgramData" -Name | Select-String "anaconda"
Get-ChildItem "$env:USERPROFILE\anaconda*" -Name
Get-ChildItem "$env:USERPROFILE\miniconda*" -Name
```

**Solution 2: Check Program Files**
```powershell
Get-ChildItem "C:\Program Files\Anaconda3" -Name
Get-ChildItem "C:\Program Files (x86)" -Name | Select-String "anaconda"
```

**Solution 3: Use Python directly**
```powershell
# If you have Python installed elsewhere:
python --version
python -m pip install jupyter notebook
python -m jupyter notebook
```

### **Problem: "Permission Denied" or "Access Denied"**

**Solution:**
```powershell
# Run PowerShell as Administrator
# Right-click → "Run as Administrator"
# Then run the command
```

### **Problem: Jupyter starts but notebook won't open**

**Solution:**
```powershell
# Manually open your browser and go to:
http://localhost:8888

# Or use this token from the console:
# Copy the URL that appears in terminal
```

### **Problem: Still not working after all this?**

**Fallback: Run the direct comparison instead**
```batch
RUN_WITH_ANACONDA.bat
```

This doesn't need Jupyter and will complete in 3-5 minutes.

---

## 📊 Alternative: Use the Direct Comparison Script

Don't want Jupyter? No problem! Just run:

```batch
RUN_WITH_ANACONDA.bat
```

This executes `RL_Algorithms_Comparison.py` directly and gives you:
- ✅ Full training of all 3 algorithms
- ✅ Performance metrics
- ✅ Visualizations
- ✅ Recommendations
- ✅ **No Jupyter needed!**

**Time: 3-5 minutes**

---

## 🎓 What Each File Does

### **RUN_WITH_ANACONDA.bat**
- Uses Anaconda Python from `C:\ProgramData\anaconda3.1`
- Installs torch, numpy, matplotlib, seaborn, pandas
- Runs `RL_Algorithms_Comparison.py` directly
- **No Jupyter required**
- **Time: 3-5 minutes**
- **Best for: Getting fast results**

### **LAUNCH_JUPYTER.bat**
- Uses Anaconda Python from `C:\ProgramData\anaconda3.1`
- Installs Jupyter and dependencies
- Opens `RL_Algorithm_Comparison_NeuroFire.ipynb` in browser
- **Requires Jupyter installation**
- **Time: 5-10 minutes first run, then interactive**
- **Best for: Learning and exploration**

### **LAUNCH_JUPYTER.ps1**
- PowerShell version of the batch file
- Same functionality as LAUNCH_JUPYTER.bat
- **For users who prefer PowerShell**

---

## 🚀 RECOMMENDED: FASTEST PATH

If you just want results with no complications:

### **Step 1: Open Command Prompt**
```
Windows key → cmd → Enter
```

### **Step 2: Navigate to folder**
```cmd
cd "C:\Users\Edam\Downloads\RL\NeuroFire"
```

### **Step 3: Run this command**
```cmd
C:\ProgramData\anaconda3.1\python.exe RL_Algorithms_Comparison.py
```

### **Step 4: Wait 3-5 minutes**
Watch the training progress and metrics!

### **Step 5: Results**
- `neurofire_rl_comparison.png` (main visualization)
- Console metrics and recommendations

---

## ✨ Success Criteria

You'll know it's working when you see:

### **For Direct Comparison:**
```
Training DQN...
Episode 200 | Reward: 15.32 | Mean: 12.45

Training PPO...
Episode 200 | Reward: 18.92 | Mean: 13.92

Training A2C...
Episode 200 | Reward: 14.21 | Mean: 10.33

Algorithm | Mean Reward | Std Dev
----------|-------------|--------
PPO       |    13.92    |  2.10  ⭐
```

### **For Jupyter Notebook:**
```
JupyterLab 4.x.x is running at:
    http://localhost:8888/?token=...

Shutting down (Press CTRL+C to exit)
```

---

## 📞 Last Resort: Manual Fix

If nothing above works, try this complete reset:

```powershell
# 1. Uninstall old Jupyter (if installed)
C:\ProgramData\anaconda3.1\python.exe -m pip uninstall -y jupyter notebook

# 2. Install fresh
C:\ProgramData\anaconda3.1\python.exe -m pip install --upgrade jupyter notebook ipython

# 3. Verify
C:\ProgramData\anaconda3.1\python.exe -m jupyter --version

# 4. Launch
cd "C:\Users\Edam\Downloads\RL\NeuroFire"
C:\ProgramData\anaconda3.1\python.exe -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb
```

---

## 🎯 Bottom Line

### **Want quick results?**
```
👉 Run: RUN_WITH_ANACONDA.bat
⏱️  3-5 minutes
✅ No Jupyter needed
```

### **Want interactive learning?**
```
👉 Run: LAUNCH_JUPYTER.bat
⏱️  5-10 minutes setup
✅ Then explore notebook interactively
```

### **Already have Jupyter?**
```
👉 Run: C:\ProgramData\anaconda3.1\python.exe -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb
⏱️  Opens immediately
✅ Use your existing Jupyter
```

---

**Choose your method and start exploring!** 🚀
