# House Price Valuation Prototype 🏠

An end-to-end Machine Learning web application designed to evaluate and predict property valuations using **Multiple Linear Regression** and **Polynomial Regression (Degree 2)**. Built with **Python**, **scikit-learn**, and **Streamlit**, and deployed live on Streamlit Cloud.

---

## 📌 Project Overview

Valuing real estate properties accurately requires capturing both basic linear trends (such as square footage and room counts) and complex non-linear feature interactions (such as the compounded value of square footage in top-tier localities).

This project addresses this challenge by providing:
1. **Multiple Linear Regression**: Establishes a baseline linear relationship between real estate metrics and target valuations.
2. **Polynomial Regression (Degree 2)**: Introduces non-linear interaction terms to model spatial pricing curves and location dynamics accurately.
3. **Interactive Buyer Valuation Widget**: A real-time web application allowing stakeholders, buyers, and appraisers to input property specifications and immediately obtain market estimates.

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language**: Python 3.9+
- **Machine Learning & Preprocessing**: `scikit-learn`
- **Data Manipulation & Processing**: `pandas`, `numpy`
- **Web Interface**: `streamlit`
- **Deployment Platform**: Streamlit Community Cloud
- **Version Control**: Git / GitHub

---

## 📊 Dataset & Feature Definitions

The system models simulated municipal property registry data containing structural and contextual valuation features:

| Feature Name | Feature Type | Description |
| :--- | :--- | :--- |
| `SquareFeet` | Continuous | Total covered built-up area of the property (sq ft) |
| `Bedrooms` | Discrete | Total number of bedrooms |
| `Bathrooms` | Discrete | Total number of bathrooms |
| `LocalityScore` | Ordinal/Continuous | Index score (1.0 to 10.0) evaluating safety, access, and infrastructure |
| `AmenitiesScore` | Continuous | Index score (1.0 to 10.0) reflecting proximity to schools, transit, and parks |
| `Price` *(Target)* | Continuous | Estimated market valuation ($) |

---

## 📐 Mathematical Methodology

### 1. Multiple Linear Regression
Models property valuation as a linear combination of input features:
$$\text{Price} = \beta_0 + \beta_1(\text{SqFt}) + \beta_2(\text{Beds}) + \beta_3(\text{Baths}) + \beta_4(\text{Locality}) + \beta_5(\text{Amenities}) + \epsilon$$

### 2. Polynomial Regression ($Degree = 2$)
Captures feature interactions and non-linearities using a second-degree polynomial transformation:
$$\text{Price} = \beta_0 + \sum_{i=1}^{n} \beta_i X_i + \sum_{i=1}^{n} \beta_{ii} X_i^2 + \sum_{i < j} \beta_{ij} X_i X_j + \epsilon$$

---

## 🚀 How to Run Locally

### Prerequisites
Ensure Python 3.8+ is installed on your computer.

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/house-valuation-app.git
   cd house-valuation-app
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**:
   ```bash
   python -m streamlit run app.py
   ```

---

## 📁 Repository Structure

```text
├── app.py              # Main Python script containing data pipeline, ML logic & UI
├── requirements.txt    # Python dependencies list for cloud deployment
└── README.md           # Detailed project documentation
```

---

## 🌐 Live Web Application

The live interactive application is deployed and accessible online:  
👉 **[House Price Valuation Prototype Live Demo](https://house-valuation-app-xtwtmwcx7o23dn3yqm2va3.streamlit.app/)**
