# Federated Learning for Fertility Risk Prediction

A privacy-preserving federated learning system for predicting fertility risks using the Flower framework and PyTorch.

## 🎯 Project Overview

This project implements a federated learning system where multiple hospitals can collaboratively train a machine learning model for fertility risk prediction without sharing patient data.

**Key Features:**
- ✅ Privacy-preserving federated learning
- ✅ Real-world medical dataset (5.2M patient records)
- ✅ 99.2% prediction accuracy
- ✅ Differential privacy implementation
- ✅ Multi-hospital simulation (5 clients)

## 📊 Dataset

**Source:** Annual Health Survey (AHS) India - Woman Dataset
- **Download from:** [Kaggle - AHS Woman Dataset](https://www.kaggle.com/datasets/rajanand/ahs-woman-1)
- **Size:** 5.2 million patient records
- **Features:** 34 fertility-related features
- **Target:** Binary classification (Low Risk / High Risk)

**Note:** Dataset is NOT included in this repository due to size (3.4GB) and privacy. Download separately and place in `data/raw/`.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 16GB RAM (for running 5 clients)
- ~10GB free disk space (for dataset)

### Installation

1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/fertility-fl-project.git
cd fertility-fl-project
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download dataset:
   - Download from [Kaggle link]
   - Extract CSV files to `data/raw/`

5. Prepare data:
```bash
python prepare_data.py
```

6. Run federated learning:
```bash
flwr run
```

## 📁 Project Structure
```
fertility-fl-project/
├── fertility_fl/          # Main package
│   ├── client_app.py      # FL client (no DP)
│   ├── client_app_dp.py   # FL client (with DP)
│   ├── server_app.py      # FL server
│   ├── model.py           # Neural network
│   └── task.py            # Data loading
├── prepare_data.py        # Data preprocessing
├── pyproject.toml         # Flower configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🧠 Model Architecture

- **Type:** Deep Neural Network
- **Input:** 34 features
- **Hidden Layers:** 128 → 64 → 32 neurons
- **Output:** 2 classes (Low/High Risk)
- **Parameters:** ~180,000

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.23% |
| Training Time | 81 minutes (10 rounds) |
| Clients | 5 hospitals |
| Privacy | Federated (data never shared) |

## 🔒 Privacy Features

- **Federated Learning:** Raw data never leaves hospitals
- **Differential Privacy:** Optional ε-DP implementation
- **Secure Aggregation:** Only model updates shared

## 🛠️ Technologies

- **Federated Learning:** Flower Framework
- **Deep Learning:** PyTorch
- **Privacy:** Opacus (Differential Privacy)
- **Data Processing:** NumPy, Pandas, Scikit-learn

## 📚 Citation

If you use this code, please cite:
```bibtex
@misc{fertility_fl_2025,
  title={Federated Learning for Fertility Risk Prediction},
  author={Chandana N C, Drupitha C, Keerthana S},
  year={2025},
  institution={M S Ramaiah Institute of Technology}
}
```

## 👥 Authors

- Chandana N C (1MS22CI018)
- Drupitha C (1MS22CI023)
- Keerthana S (1MS22CI034)

**Guided by:** Dr. Naveen N C

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Annual Health Survey (AHS) India for the dataset
- Flower team for the federated learning framework
- Dr. Naveen N C for guidance

## 📧 Contact

For questions or collaborations, please open an issue or contact:
- Email: [your-email@example.com]
- GitHub: [@your-username]