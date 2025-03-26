**Overview**
This project is a Diabetes Prediction Model built using the Random Forest algorithm to classify individuals as diabetic or non-diabetic based on key health parameters. The model is deployed using FastAPI, allowing real-time predictions via a web-based API.

**Features-**
 Machine Learning Model – Uses Random Forest for reliable predictions
 FastAPI Deployment – Lightweight, high-performance API integration
 Real-Time Predictions – Accepts user inputs and provides instant results
 RESTful API – Accessible via HTTP requests for seamless integration

**Tech Stack**
Python (pandas, numpy, scikit-learn)

FastAPI (for API development)

Uvicorn (for running the API server)

Docker (optional, for containerization)

**API Endpoint**
🔹 POST /predict

Input: JSON with health parameters (e.g., glucose level, BMI, age)

Output: Diabetes prediction (0 or 1) with probability
