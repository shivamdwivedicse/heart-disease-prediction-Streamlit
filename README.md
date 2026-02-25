# heart-disease-prediction-Streamlit
This project is an end-to-end Machine Learning web application built using **Python and Streamlit** to predict the risk of heart disease based on patient health data.

Model training notebook available here:- https://github.com/shivamdwivedicse/Machine-Learning-Repo/blob/main/Heart_disease.ipynb

## 🚀 Features
- Data preprocessing (encoding & scaling)
- Machine Learning models: KNN 
- Model persistence using Pickle (joblib)
- Interactive web interface using Streamlit
- Probability-based prediction (Logistic Regression)

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## 📂 Project Structure
heart-disease-prediction-streamlit/
│
├── app.py
├── KNN_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md

This project is for educational purposes and demonstrates an end-to-end ML workflow.

📊 Model Details:-
1.KNN for final prediction.
2.StandardScaler for feature scaling.
3.One-hot encoding for categorical variables.

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

