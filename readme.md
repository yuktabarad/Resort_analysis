# Hotel Booking Cancellation Prediction

## Overview
This project uses machine learning to predict whether a hotel booking will be canceled based on booking and guest information.

## Workflow
- Explore and visualize the dataset (EDA)
- Clean missing values and duplicates
- Encode categorical variables and scale features where needed
- Engineer useful features
- Train classification models such as Logistic Regression, Decision Tree, and Random Forest
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix

## Target
`is_canceled` — indicates whether a booking was canceled.

## Important
Exclude `reservation_status` because it reveals the booking outcome and can cause target leakage. Use only features that would be available at the time of prediction.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
