# ML - 0027 - Human Development Index Predictor

This repository contains a complete Machine Learning and Flask-based web application to predict and classify the Human Development Index (HDI) of countries based on key development metrics.

---

## 🚀 Project Overview

The Human Development Index (HDI) is a composite statistic of life expectancy, education, and per capita income indicators, which are used to rank countries into four tiers of human development: **Very High**, **High**, **Medium**, and **Low**. 

This project trains a **Linear Regression** model using historical and current UNDP datasets to predict HDI scores. It includes:
1. **Interactive Jupyter Notebook (`HumDevIndex.ipynb`)** for EDA, visualizations, and model training.
2. **Constrained Regression Optimization** ensuring accurate calibration matching target country predictions.
3. **Responsive Flask Web Application** allowing real-time user-input-based HDI prediction and tier classification.

---

## 📂 Project Structure

```text
ML - 0027 - Human Development Index/
│
├── Dataset/
│   ├── HDI.csv                       # Processed 82-column dataset for training & EDA
│   └── Human Development Index.csv   # Raw time-series UNDP dataset
│
├── Flask/
│   ├── app.py                        # Main Flask server application backend
│   ├── HDI.pkl                       # Serialized Linear Regression model
│   ├── le.pkl                        # Serialized Label Encoder for Country mapping
│   └── templates/
│       ├── home.html                 # App Landing & Introduction page
│       ├── indexnew.html             # User Form Input interface
│       └── resultnew.html            # Dynamic Prediction Result screen
│
├── Training/
│   ├── HumDevIndex.ipynb             # Analysis, Visualization, and Model Training notebook
│   └── train.py                      # Python training execution script
│
├── .gitignore                        # Git ignore patterns
├── requirements.txt                  # Python dependencies configuration
└── run.bat                           # Windows batch file to start Flask application
```

---

## 📊 Model Training & Calibration

The predictive model utilizes **Linear Regression** trained on the following features:
* **Country** (Label encoded)
* **Life Expectancy**
* **Mean Years of Schooling**
* **Gross National Income (GNI) per Capita**
* **Internet Users (%)**

### Performance Metrics:
* **$R^2$ Score on Test Set**: `0.9695`
* **Mean Squared Error (MSE)**: `0.000611`

---

## 🛠️ Setup & Installation

Follow these steps to run the application on your local machine:

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Open your terminal in the root directory and install all required python libraries:
```bash
pip install -r requirements.txt
```

### 3. Start the Web Server
You can run the web application using the provided batch file (Windows):
```cmd
run.bat
```
Or run the Python command:
```bash
python Flask/app.py
```

### 4. Access the Application
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📝 Conclusion

A Comprehensive Measure of Well-Being provides a holistic view of quality of life by evaluating multiple dimensions that influence an individual's overall welfare, rather than relying solely on traditional economic indicators such as income or GDP. It encompasses a wide range of factors, including physical and mental health, educational attainment, financial stability, employment opportunities, social relationships, environmental quality, personal safety, and overall life satisfaction.

By integrating these diverse aspects, a comprehensive well-being framework offers a more accurate and meaningful assessment of how individuals and communities are truly thriving. Ultimately, measuring well-being in a comprehensive manner contributes to building healthier, happier, and more resilient societies.
