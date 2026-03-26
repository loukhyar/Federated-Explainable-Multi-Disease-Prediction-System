# 🩺 Federated Explainable Multi-Disease Prediction System

A privacy-aware, AI-powered healthcare prediction platform that supports **multi-disease risk assessment** with **interpretable explanations** using SHAP and LIME.

---

## 🚀 Live Demo

👉 *Add your deployed link here after deployment*

---

## 🧠 Problem Statement

Healthcare prediction systems often:

* lack transparency in decision-making
* require centralized sensitive patient data
* are difficult for non-technical users to interpret

This project addresses these issues by combining:

* **Explainable AI (XAI)**
* **Simulated Federated Learning**
* **Multi-disease prediction models**

---

## 💡 Key Features

* 🧠 **Multi-Disease Prediction**

  * Diabetes
  * Heart Disease
  * Lung Disease
  * Liver Disease

* 🔍 **Explainable AI**

  * SHAP (global + local explanations)
  * LIME (instance-level interpretation)
  * Human-readable explanations for non-technical users

* 🔐 **Privacy-Aware Design**

  * Simulated federated learning across datasets
  * Avoids centralizing sensitive healthcare data

* 📊 **Interactive UI**

  * Built using Streamlit
  * Clean and user-friendly interface
  * Real-time prediction + explanation

---

## 🏗️ System Architecture

```
User Input → Model Selection → Prediction Engine → 
Explainability Layer (SHAP + LIME) → Output + Insights
```

Federated Learning (Simulated):

* Multiple datasets represent distributed hospitals
* Models trained independently and integrated

---

## ⚙️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **ML Models**: Scikit-learn
* **Explainability**: SHAP, LIME
* **Data Handling**: Pandas, NumPy
* **Model Storage**: Joblib

---

## 📂 Project Structure

```
healthfed/
│── app.py
│── data/
│── models/
│── pages/
│── training/
│── requirements.txt
```

---

## 🧪 How It Works

1. User selects a disease
2. Inputs patient data
3. Model predicts risk probability
4. System generates:

   * Risk level (Low / Moderate / High)
   * SHAP visualization
   * LIME explanation
   * Human-readable reasoning

---

## 🔍 Explainability Example

Instead of just predicting risk, the system explains:

> “High glucose level and BMI increased the risk of diabetes.”

This makes predictions:

* understandable
* transparent
* actionable

---

## 🏥 Medical Disclaimer

This system is for **educational and research purposes only**.
It should not be used as a substitute for professional medical advice.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/loukhyar/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 🚀 Deployment

Recommended:

* Streamlit Cloud (fastest)
* Render (better production feel)

---

## 📈 Future Improvements

* Real federated learning using distributed clients
* Integration with real-time healthcare APIs
* Improved UI (React-based frontend)
* Model performance optimization

---

## 👤 Author

**Loukhya Reddy Thatikonda**

* GitHub: https://github.com/loukhyar
* LinkedIn: https://linkedin.com/in/loukhya

---

## ⭐ Why This Project Stands Out

* Combines **ML + System Design + Explainability**
* Focuses on **real-world healthcare challenges**
* Goes beyond prediction to **decision transparency**

---
