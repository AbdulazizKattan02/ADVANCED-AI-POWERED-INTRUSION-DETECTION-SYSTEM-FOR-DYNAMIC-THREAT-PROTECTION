# ADVANCED-AI-POWERED-INTRUSION-DETECTION-SYSTEM-FOR-DYNAMIC-THREAT-PROTECTION
=======

# Network Traffic Classifier

This application uses machine learning to classify network traffic as benign or harmful, and identifies the type of attack if harmful. The system provides explanations for its decisions using SHAP and LIME explainability techniques.

## Features

- Three different machine learning models: GBM, LightGBM, and XGBoost
- Multiple input methods: Manual input, CSV upload, and sample data
- Detailed explanations of predictions using SHAP and LIME
- Visualization of feature importance
- Recommendations based on classification results

## How to Use

1. Select a model from the sidebar (LightGBM recommended)
2. Choose an input method:
   - Manual Input: Enter network traffic features manually
   - Upload CSV: Upload a CSV file with network traffic data
   - Sample Data: Use pre-defined sample data for different traffic types
3. Click "Classify Traffic" to get results
4. View the classification results, explanations, and recommendations

## Models

- **GBM (Gradient Boosting Machine)**: A traditional gradient boosting implementation
- **LightGBM**: Uses a leaf-wise tree growth strategy, faster and often more accurate
- **XGBoost**: Uses regularized model formalization to control overfitting

## Explainability

- **SHAP (SHapley Additive exPlanations)**: Shows how each feature contributes to the prediction
- **LIME (Local Interpretable Model-agnostic Explanations)**: Provides interpretable explanations for individual predictions

## Attack Types

- **Denial of Service (DoS)**: Overwhelming network resources to make them unavailable
- **Probing/Scanning**: Scanning network ports and services to identify vulnerabilities
- **Remote to Local (R2L)**: Unauthorized access from a remote machine
- **User to Root (U2R)**: Escalation of privileges from user to root/admin
