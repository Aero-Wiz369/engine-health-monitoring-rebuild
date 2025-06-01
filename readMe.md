# ✈️ Engine Health Monitoring using CMAPSS Data

This project trains a machine learning model to **predict the Remaining Useful Life (RUL)** of commercial jet engines using the **NASA CMAPSS dataset (FD001)**.

---

## 📦 Project Structure

engine_health_monitoring/
├── data/
│ └── FD001.txt
├── models/
│ └── random_forest_model.pkl ← (Downloadable below)
├── notebooks/
│ └── EDA_and_Insights.ipynb
├── src/
│ ├── data_preprocessing.py
│ └── model_training.py
├── main.py
├── requirements.txt
└── README.md


---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| MAE    | `29.69` |
| R²     | `0.62` |

Model used: **Random Forest Regressor**

---

## 🔍 Dataset Used

- **NASA CMAPSS (FD001 subset)**
- Contains simulated degradation data of jet engines
- Each engine has sensor readings across multiple cycles until failure

Download CMAPSS: [NASA CMAPSS Link](https://www.nasa.gov/sites/default/files/atoms/files/cmapps_data.zip)

---

## 🛠️ How to Run This Project

### 1. Clone the Repo

```bash
git clone https://github.com/Aero-Wiz369/engine-health-monitoring-rebuild.git
cd engine-health-monitoring-rebuild

INSTALL REQUIREMENTS
pip install -r requirements.txt

Download trained modelvat the following link:
https://drive.google.com/file/d/1LLTusYi84pEm-M2jigQ8rTxKxPJWF_pK/view?usp=sharing

