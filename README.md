# Disease Prediction System

## Project Components

- **Machine Learning Model**  
  A classification system that predicts diseases based on symptom descriptions. It uses Random Forest, Decision Tree, and Naive Bayes algorithms, with an automatic process to choose the best-performing model.

- **Backend API**  
  A Flask-based REST API that serves the trained model. It includes endpoints for checking system health, retrieving symptoms, and predicting diseases.

- **Frontend Interface**  
  A responsive web application where users can select symptoms, search in real time, and view predictions.

- **Dataset**  
  The system is trained on a Kaggle dataset of 1,200 symptom–disease pairs.  
  Source: [Symptom2Disease Dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

---

## Implementation Timeline

### Phase 1: Data Processing & Model Development

**i. Data Acquisition & Preprocessing**
- Collected and integrated the Kaggle dataset (1,200 samples, 1,153 unique diseases).
- Designed a pipeline that automatically detects input formats.
- Converted symptom text into 37 binary features.
- Used RandomOverSampler to fix class imbalance (~4:1 imbalance ratio).
- Added missing value handling.

**ii. Model Training & Selection**
- Built a comparison framework with three algorithms:
  - Random Forest (n_estimators=100)
  - Decision Tree
  - Gaussian Naive Bayes
- Used adaptive cross-validation:
  - KFold (3-splits) for datasets with more than 100 classes
  - StratifiedKFold (5-splits) for balanced data
- The system automatically picks the best model based on performance.
- Added feature importance analysis to understand model decisions.

**iii. Backend API**
- Flask REST API with three endpoints:
  - `/health` → system status and metrics
  - `/symptoms` → list of available symptoms
  - `/predict` → prediction with confidence scores
- Added CORS support and proper error handling.
- Model persistence using joblib.

---

### Phase 2: Frontend & Integration

**i. Frontend Development**
- Responsive HTML5/CSS3/JS web app.
- Key features:
  - Real-time search with autocomplete
  - Interactive checkbox-based symptom selection
  - Visual tags for selected symptoms
  - API connection monitoring
  - Mobile-friendly layout

**ii. API Integration**
- Built automatic health checks and fallback mechanisms.
- Implemented async requests with loading indicators.
- Two modes:
  - **Connected mode** → live ML predictions
  - **Demo mode** → offline fallback for presentations
- Added user-friendly error messages.

**iii. Testing & Validation**
- Tested end-to-end with various symptom combinations.
- Validated accuracy and confidence scores.
- Covered edge cases:
  - Network issues
  - Invalid inputs
  - API downtime
- Added disclaimers and usage guidelines.

---

## Technical Overview

- **Algorithm:** Ensemble with auto-selection  
- **Features:** 37 binary symptom features  
- **Class Handling:** RandomOverSampler  
- **Validation:** Adaptive cross-validation  
- **Metrics:** Accuracy, precision, recall, F1-score  

- **API:** Flask + CORS, joblib persistence, JSON responses, error handling  
- **Frontend:** HTML5, CSS3 (Flexbox/Grid), vanilla JS, responsive design  
- **Data Processing:** Text input → multi-class prediction with ranked confidence  

---

## Achievements

1. Successfully trained a model on 1,153 disease classes with auto algorithm selection.  
2. Built a production-ready Flask API with robust error handling.  
3. Designed a simple, user-friendly interface with real-time feedback.  
4. Developed a flexible data pipeline that adapts to input variations.  
5. End-to-end system tested and ready for deployment.  

---
## Project Structure

disease-prediction/
├── disease_predictor.py # ML training pipeline
├── flask_api.py # REST API server
├── index.html # Web frontend
├── requirements.txt # Dependencies
├── symptom2disease.xlsx # Dataset
├── disease_model.joblib # Trained model
├── model_metadata.json # Model details
└── README.md # Documentation


---


## Performance

- Dataset expanded from 1,200 → 4,612 samples after balancing  
- 37 binary features extracted  
- Model accuracy improved with cross-validation  
- API response time <100ms  
- Frontend real-time responsiveness achieved  
- Reliable system with fallback mechanisms  

---

## Future Enhancements

- Symptom severity scoring  
- Multi-label predictions  
- Explainable AI (e.g., SHAP values)  
- Categorized symptoms & patient history integration  
- Containerized deployment (Docker, Cloud)  
- Performance monitoring dashboard  

---

## Medical & Ethical Notes

- System is for **educational purposes only**.  
- Predictions are not a substitute for medical advice.  
- Clear disclaimers included.  
- Encourages professional consultation.  


