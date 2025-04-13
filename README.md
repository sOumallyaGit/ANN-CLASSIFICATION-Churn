# 🔍 Customer Churn Prediction

This project predicts whether a customer is likely to **churn** (leave a company or service) using an **Artificial Neural Network (ANN)**. It also features an **interactive Streamlit web app** for real-time predictions based on user input.

---

## ❓ Problem Statement

Customer churn is a major issue across industries like telecom, banking, and SaaS. Losing existing customers is often more expensive than acquiring new ones. The goal of this project is to:

> **Build a deep learning model** that analyzes historical customer data and identifies those who are likely to leave, enabling companies to take timely, proactive retention actions.

The app predicts churn based on key features like **tenure, contract type, monthly charges, and usage patterns**.

---

## 📌 Key Features

- Clean and preprocess customer data
- Explore the dataset with visual insights (EDA)
- Build and train an **Artificial Neural Network (ANN)** using TensorFlow/Keras
- Evaluate the ANN model using standard classification metrics
- Predict churn on new customer data
- Deploy predictions using a **Streamlit web application**

---

## ⚙️ ANN Model Details

- Input Layer: Accepts numerical and one-hot encoded categorical features  
- Hidden Layers: Multiple fully connected layers using ReLU activation  
- Output Layer: Single neuron with sigmoid activation for binary classification  
- Loss Function: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC  

The model is trained to classify whether a customer will churn (1) or stay (0).

---

## 🚀 Streamlit Web App

The project includes a deployed [Streamlit app](https://ann-classification-churn-q9yntdl9yq336mobkwmpnm.streamlit.app/)

### 🔗 Access the App:
👉 [Click here to open the live demo](https://ann-classification-churn-q9yntdl9yq336mobkwmpnm.streamlit.app/)

### Features:

- Simple UI for entering customer details
- Instant prediction with model confidence score
- Visual feedback on whether the customer is likely to churn

### Run it locally:

```bash
streamlit run app/app.py
