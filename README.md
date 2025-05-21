# Sales Forecasting and Store Analytics Project

## Overview

This project aims to forecast sales and analyze store performance using advanced machine learning and deep learning techniques. The workflow covers data ingestion, exploratory data analysis (EDA), feature engineering, model training (including classical ML and deep learning models), experiment tracking, and deployment of the best model for inference.

---

## Project Structure

```
Database/
    Database Insertion Process.ipynb
    Store_Sales_db.sql
    training_data_insertion.py
    data/
Delopyment/
    app.py
    best_model.pkl
    family_label_encoder.pkl
    x_scaler.pkl
    y_scaler.pkl
EDA/
    app.py
    Sales_prediction.ipynb
    sales-forcasting-depi.ipynb
Modeling/
    Experiments.ipynb
    GradientBoost Classification Implementation.ipynb
    Model.ipynb
    Sales_forecasting.ipynb
    mlruns/
```

---

## Components

### 1. Database

- **Store_Sales_db.sql**: SQL schema for the sales database.
- **training_data_insertion.py**: Script to insert training data into the database.
- **Database Insertion Process.ipynb**: Notebook documenting the data ingestion process.

### 2. EDA

- **Sales_prediction.ipynb** & **sales-forcasting-depi.ipynb**: Notebooks for exploratory data analysis, feature engineering, and initial modeling.
- **requirements.txt**: Python dependencies for EDA and modeling.
- **app.py**: (Optional) EDA dashboard or utility scripts.

### 3. Modeling

- **Sales_forecasting.ipynb**: Main notebook for sales forecasting using XGBoost and MLFlow for experiment tracking.
- **GradientBoost Classification Implementation.ipynb**: Notebook for classification tasks using Gradient Boosting.
- **Model.ipynb**: Deep learning models (RNN, LSTM, GRU) for time series forecasting, including MLFlow logging.
- **Experiments.ipynb**: Additional experiments and model comparisons.
- **mlruns/**: MLFlow tracking directory for experiment metadata and artifacts.
- **best_model.pkl**: Serialized best-performing model.

### 4. Deployment

- **app.py**: Streamlit app for serving the trained model.
- **best_model.pkl**, **family_label_encoder.pkl**, **x_scaler.pkl**, **y_scaler.pkl**: Model and preprocessing artifacts for inference.

---

## Key Features

- **Data Engineering**: SQL schema and scripts for structured data storage.
- **EDA**: In-depth analysis and visualization of sales data.
- **Modeling**: 
  - Classical ML (XGBoost, Gradient Boosting)
  - Deep Learning (RNN, LSTM, GRU with Keras)
- **Experiment Tracking**: MLFlow integration for reproducibility and model management.
- **Deployment**: Ready-to-use API for predictions.

---

## How to Run

1. **Set up the Database**
   - Use `Store_Sales_db.sql` to create the database schema.
   - Run `training_data_insertion.py` to populate the database.

2. **Install Dependencies**
   - Navigate to the EDA or Modeling folder and run:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run EDA and Modeling Notebooks**
   - Open notebooks in the `EDA/` and `Modeling/` folders for analysis and training.

4. **Track Experiments**
   - MLFlow is used for experiment tracking. Start the MLFlow server:
     ```sh
     mlflow ui
     ```
   - Access the UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

5. **Deploy the Model**
   - Use the `Delopyment/app.py` script to serve the trained model via an API.

---

## Results

- **Best Model**: XGBoost and deep learning models are compared; the best model is saved as `best_model.pkl`.
- **Metrics**: RMSE, MAE, R2 for regression; Accuracy, F1, Precision, Recall, ROC AUC for classification.
- **Experiment Tracking**: All runs and parameters are logged in MLFlow.

---

## Authors

- [Mohamed Khalf](https://github.com/Mohamed-khalf30)
- [Mahmoud Mansour](https://github.com/MahmoudMansour27)
- [Mina William](https://github.com/mina4747)
- [Mohamed Walid]()
