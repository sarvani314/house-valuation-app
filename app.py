import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(page_title="House Price Valuation Prototype", page_icon="🏡", layout="wide")

st.title("🏡 House Price Valuation Prototype")
st.markdown("### Multiple Linear & Polynomial Regression Model for Property Valuation")
st.markdown("*Trained and validated using Municipal Property Registry Data with a buyer-facing valuation widget.*")

# 1. Dataset Generation & Preprocessing (Simulating Municipal Registry Data)
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1200
    
    sqft = np.random.randint(600, 4500, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.randint(1, 5, size=n_samples)
    locality_score = np.random.randint(1, 6, size=n_samples)
    amenities_score = np.random.randint(1, 11, size=n_samples)
    
    # Simulating non-linear price behavior for polynomial fitting
    base_price = 400000
    price = (
        base_price 
        + (sqft * 2500) 
        + (bedrooms * 150000) 
        + (bathrooms * 120000) 
        + (locality_score * 350000) 
        + (amenities_score * 100000)
        + (sqft * locality_score * 150) # Feature interaction
        + np.random.normal(0, 200000, size=n_samples)
    )
    
    df = pd.DataFrame({
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'LocalityScore': locality_score,
        'AmenitiesScore': amenities_score,
        'Price': price
    })
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation Panel")
app_mode = st.sidebar.selectbox("Choose Section", ["1. Dataset & Preprocessing", "2. Model Training & Comparison", "3. Buyer Valuation Widget"])

# Train-Test Split & Modeling
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'LocalityScore', 'AmenitiesScore']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Polynomial Regression (Degree 2)
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

# Section 1: Dataset & EDA
if app_mode == "1. Dataset & Preprocessing":
    st.header("📊 Municipal Property Registry Data & Preprocessing")
    st.write("Preview of cleaned records featuring property dimensions, locality indices, and amenities scores:")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical Summary")
        st.write(df.describe())
    with col2:
        st.subheader("Feature Correlation with Price")
        st.bar_chart(df.corr()['Price'].drop('Price'))

# Section 2: Model Comparison
elif app_mode == "2. Model Training & Comparison":
    st.header("⚙️ Model Performance Evaluation")
    st.write("Comparing Multiple Linear Regression against Polynomial Regression (Degree 2) to capture non-linear market trends.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Multiple Linear Regression")
        r2_lin = r2_score(y_test, y_pred_lin)
        rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
        st.metric(label="R² Score", value=f"{r2_lin:.4f}")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"₹ {rmse_lin:,.2f}")
        st.write("Assumes a direct additive relationship between features and price.")

    with col2:
        st.subheader("Polynomial Regression (Degree 2)")
        r2_poly = r2_score(y_test, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
        st.metric(label="R² Score", value=f"{r2_poly:.4f}")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"₹ {rmse_poly:,.2f}")
        st.write("Accounts for complex interaction terms like size combined with premium locality ratings.")

# Section 3: Valuation Widget
elif app_mode == "3. Buyer Valuation Widget":
    st.header("🏷️ Buyer-Facing Valuation Widget")
    st.markdown("Use this interactive tool to estimate property prices based on real-time feature inputs.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sqft_input = st.number_input("Square Footage (SqFt)", min_value=500, max_value=5000, value=1800, step=50)
        bed_input = st.slider("Bedrooms", 1, 5, 3)
        bath_input = st.slider("Bathrooms", 1, 4, 2)
        
    with col2:
        locality_input = st.slider("Locality Score (1 = Standard, 5 = Prime)", 1, 5, 4)
        amenities_input = st.slider("Amenities Score (1 to 10)", 1, 10, 8)
        model_choice = st.selectbox("Select Valuation Algorithm", ["Polynomial Regression (Recommended)", "Multiple Linear Regression"])

    input_data = pd.DataFrame([[sqft_input, bed_input, bath_input, locality_input, amenities_input]], 
                              columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'LocalityScore', 'AmenitiesScore'])
    
    if st.button("Calculate Market Valuation", type="primary"):
        if "Polynomial" in model_choice:
            prediction = poly_pipeline.predict(input_data)[0]
        else:
            prediction = lin_reg.predict(input_data)[0]
            
        st.success(f"### Estimated Property Price: ₹ {prediction:,.2f}")
        st.balloons()