# House Price Predictor

A Flask web application that predicts residential property prices in Bengaluru (Bangalore) based on location, square footage, number of bedrooms (BHK) and bathrooms—powered by a scikit-learn Ridge regression pipeline.

---

## 🔍 Project Overview

- **Data Cleaning & Feature Engineering**  
  - Converts messy `total_sqft` ranges into numeric values  
  - Derives `bhk` from the `size` column  
  - Groups rare locations into “other”  
  - Removes outliers based on sqft per BHK and price per sqft  

- **Modeling**  
  - One-Hot Encoding for categorical `location`  
  - Standard scaling of numeric features  
  - Ridge regression to balance bias/variance  
  - Evaluated via Train/Test R² (typically ~0.8–0.9 on this dataset)

- **Web App**  
  - Bootstrap-based form (`templates/index.html`)  
  - AJAX-style submission with JavaScript `fetch`  
  - Serves predictions in ₹ (rupees) via a `/predict` endpoint  
  - Deployed to Render.com for free hosting

---

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/house-price-predictor.git
   cd house-price-predictor

