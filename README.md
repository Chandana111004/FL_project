# 🌸 Fertility Risk Prediction using Federated Learning

A privacy-preserving federated learning system for predicting fertility risk across **4 categories** using the NFHS-5 (India) dataset, Flower framework, and PyTorch — without sharing raw patient data between hospitals.

---

## 🎯 Project Overview

Five simulated hospital clients collaboratively train a Deep Neural Network model using **Federated Learning** and **Differential Privacy**. Raw patient data never leaves any hospital — only privacy-protected model updates are shared with the central server.

**Key Features:**
- ✅ Federated Learning — 5 hospital clients, zero data sharing
- ✅ Differential Privacy — Opacus library, ε = 0.15 / 5.0
- ✅ 4-Class Risk Prediction — No Risk / Low Risk / High Risk / Critical
- ✅ Real-world NFHS-5 dataset — 724,115 women
- ✅ AES-256 Encryption — all data and model files encrypted
- ✅ Role-Based Access Control — hospital data isolation
- ✅ Audit Logging — all access recorded
- ✅ Expiring Tokens — 1-hour token expiry
- ✅ Side Channel Protection — constant-time operations
- ✅ Streamlit Prediction Dashboard

---

## 📊 Dataset

**Source:** NFHS-5 (National Family Health Survey 2019-21) — India DHS Program

- **Download from:** [DHS Program](https://dhsprogram.com/data/dataset/India_Standard-DHS_2019-21.cfm)
- **File needed:** IAIR7EFL.DTA (Individual Recode — Women's data)
- **Size:** 724,115 women aged 15-49
- **Features:** 30 clinically relevant features
- **Classes:** 4 risk categories

**Note:** Dataset is NOT included due to size (5.2GB) and privacy regulations. Register at DHS Program website to download.

**Features used:**
```
Demographics  → age, residence, education, wealth, marital status
Fertility     → total children, pregnancies, ANC visits
Contraception → method, unmet need, fertility preference
Medical       → systolic BP, diastolic BP, hemoglobin, anemia, BMI
Conditions    → diabetes, hypertension
Lifestyle     → smoking, tobacco, alcohol
Socioeconomic → insurance, toilet facility, cooking fuel
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Final Test Accuracy | 77.1% |
| Training Rounds | 20 |
| Hospital Clients | 5 |
| Privacy Budget Used | ε = 0.15 / 5.0 (3%) |
| Risk Classes | 4 (No / Low / High / Critical) |
| Dataset Size | 724,115 women |
| Features | 30 medical indicators |

---

## 🧠 Model Architecture

```
Input (30 features)
    ↓
Linear(256) + GroupNorm + ReLU + Dropout(0.3)
    ↓
Linear(128) + GroupNorm + ReLU + Dropout(0.3)
    ↓
Linear(64)  + GroupNorm + ReLU + Dropout(0.3)
    ↓
Linear(32)  + GroupNorm + ReLU + Dropout(0.3)
    ↓
Output (4 classes)
```

**Why GroupNorm instead of BatchNorm?**
BatchNorm is incompatible with Opacus Differential Privacy. GroupNorm works correctly with per-sample gradients.

---

## 🔒 Security Layers

| Layer | Implementation | Details |
|-------|---------------|---------|
| Federated Learning | Flower (flwr) | Data never leaves hospitals |
| Differential Privacy | Opacus | ε = 0.15, δ = 1e-5 |
| Encryption | AES-256 (Fernet) | All data + model files |
| Access Control | RBAC | Per-hospital data isolation |
| Audit Logging | Custom | Every access recorded |
| Token Auth | JWT-style | 1-hour expiry |
| Side Channel | HMAC + timing noise | Constant-time operations |

---

## 🚀 Quick Start

### Prerequisites
```
Python 3.12
8GB+ RAM
~10GB free disk space
WSL2 (Ubuntu) recommended on Windows
```

### Installation

```bash
# Clone repository
git clone https://github.com/Chandana111004/FL_project.git
cd FL_project

# Create virtual environment
python -m venv flwr_env
source flwr_env/bin/activate  # Linux/Mac
# OR
flwr_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# Download NFHS-5 dataset from DHS Program
# Place IAIR7EFL.DTA in:
mkdir -p data/nfhs5/raw
# Copy file to data/nfhs5/raw/IAIR7EFL.DTA

# Prepare data (creates 5 client partitions + 4-class labels)
python prepare_data.py

# Run federated learning training
flwr run 2>&1 | tee results/training_log.txt

# Launch prediction dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
FL_project/
├── fertility_fl/
│   ├── __init__.py
│   ├── client_app.py        # FL client (no DP)
│   ├── client_app_dp.py     # FL client (with DP + class weights)
│   ├── server_app.py        # FL server with model saving
│   ├── model.py             # FertilityRiskNet DNN
│   ├── task.py              # Data loading utilities
│   └── security.py         # All 7 security layers
├── data/
│   └── nfhs5/raw/           # Place IAIR7EFL.DTA here
├── results/                 # Saved models + training history
├── app.py                   # Streamlit prediction dashboard
├── prepare_data.py          # NFHS-5 data preparation
├── plot_results.py          # Training results graph
├── pyproject.toml           # Flower configuration
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Configuration

Key settings in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 20
fraction-fit = 0.8
fraction-evaluate = 0.5
noise-multiplier = 0.5
max-grad-norm = 1.0
```

Key settings in `client_app_dp.py`:
```python
local_epochs = 5
```

---

## 🛠️ Technologies

| Component | Technology |
|-----------|-----------|
| Federated Learning | Flower (flwr) 1.11.1 |
| Deep Learning | PyTorch 2.1.0 |
| Differential Privacy | Opacus |
| Encryption | cryptography (Fernet AES-256) |
| Simulation | Ray |
| Dashboard | Streamlit |
| Data Processing | NumPy, Pandas, Scikit-learn, pyreadstat |

---

## 📚 Citation

```bibtex
@misc{fertility_fl_2026,
  title={Privacy-Preserving Fertility Risk Prediction using Federated Learning and Differential Privacy},
  author={Chandana N C, Drupitha C, Keerthana S},
  year={2026},
  institution={M S Ramaiah Institute of Technology}
}
```

---

## 👥 Authors

| Name | USN |
|------|-----|
| Chandana N C | 1MS22CI018 |
| Drupitha C | 1MS22CI023 |
| Keerthana S | 1MS22CI034 |

**Guided by:** Dr. Naveen N C

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- DHS Program / IIPS for the NFHS-5 dataset
- Flower team for the federated learning framework
- Meta AI for the Opacus differential privacy library
- Dr. Naveen N C for guidance and support
