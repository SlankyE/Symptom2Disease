import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptom_columns = []
        self.diseases = []
        
    def load_and_preprocess_data(self, file_path):
        print("Loading dataset...")
        
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            print("\nDataset Info:")
            print(df.head())
            print(f"\nDataset shape: {df.shape}")
            print(f"Missing values: {df.isnull().sum().sum()}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
            
        return df
    
    def prepare_features_and_labels(self, df):
        print("\n🔧 Preparing features and labels...")
        
        print(f"Dataset columns: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        
        if 'text' in df.columns and 'label' in df.columns:
            print("Detected text-based symptom format")
            
            symptom_texts = df['text'].fillna('')
            diseases = df['label']
            
            common_symptoms = [
                'fever', 'headache', 'cough', 'pain', 'nausea', 'vomiting', 'fatigue', 
                'diarrhea', 'rash', 'swelling', 'bleeding', 'weakness', 'breathing',
                'chest', 'stomach', 'throat', 'nose', 'eye', 'muscle', 'joint',
                'dizziness', 'sweating', 'chills', 'loss', 'difficulty', 'cramps',
                'itching', 'burning', 'tingling', 'numbness', 'discharge', 'spots',
                'lesions', 'ulcers', 'lumps', 'swollen', 'tender', 'stiff', 'sore'
            ]
            
            feature_matrix = []
            for text in symptom_texts:
                text_lower = text.lower()
                features = [1 if symptom in text_lower else 0 for symptom in common_symptoms]
                feature_matrix.append(features)
            
            X = pd.DataFrame(feature_matrix, columns=common_symptoms)
            y = diseases
            self.symptom_columns = common_symptoms
            
        elif 'Disease' in df.columns or 'disease' in df.columns:
            target_col = 'Disease' if 'Disease' in df.columns else 'disease'
            print(f"Detected symptom-disease format with target: {target_col}")
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            self.symptom_columns = list(X.columns)
            
        else:
            print("Auto-detecting format: assuming last column is target")
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]
            self.symptom_columns = list(X.columns)
        
        print(f"Features extracted: {len(self.symptom_columns)} symptoms")
        print(f"Target extracted: {len(y.unique())} unique diseases")
        print(f"Feature matrix shape: {X.shape}")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.diseases = list(self.label_encoder.classes_)
        
        print(f"Sample diseases: {self.diseases[:5]}...")
        print(f"Sample symptoms: {self.symptom_columns[:10]}...")
        
        return X, y_encoded
    
    def handle_class_imbalance(self, X, y):
        print("\nHandling class imbalance...")
        
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        max_samples = max(counts)
        imbalance_ratio = max_samples / min_samples
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            print(f"After resampling: {X_resampled.shape}")
            return X_resampled, y_resampled
        else:
            print("Classes are relatively balanced, no resampling needed.")
            return X, y
    
    def train_models(self, X, y):
        print("\nTraining multiple models...")
        
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        
        print(f"Number of classes: {len(unique)}")
        print(f"Minimum samples per class: {min_samples}")
        print(f"Total samples: {len(y)}")
        
        if len(unique) > len(y) * 0.3:
            test_size = max(0.1, min(0.15, (len(y) - len(unique)) / len(y)))
            print(f"Adjusted test size to: {test_size:.2f}")
        else:
            test_size = 0.2
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        if len(unique) > 100:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            print("Using KFold cross-validation (too many classes for stratified)")
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print("Using StratifiedKFold cross-validation")
        
        print("\nModel Comparison:")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                print(f"{name:15} | CV Score: {mean_score:.4f} (±{std_score:.4f})")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                print(f"{name:15} | Test Score: {test_accuracy:.4f}")
                print("-" * 50)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        print(f"\nBest Model: {best_name} with CV Score: {best_score:.4f}")
        
        self.model = best_model
        y_pred_final = best_model.predict(X_test)
        
        print(f"\nFinal Model Performance:")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
        
        unique_test = np.unique(y_test)
        if len(unique_test) > 10:
            print(f"Classification report limited to top classes (total: {len(unique_test)} diseases)")
        
        return X_test, y_test, y_pred_final
    
    def save_model(self, model_path='disease_model.joblib', metadata_path='model_metadata.json'):
        print(f"\nSaving model to {model_path}...")
        
        joblib.dump(self.model, model_path)
        
        metadata = {
            'symptom_columns': self.symptom_columns,
            'diseases': self.diseases,
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Model and metadata saved successfully!")
    
    def load_model(self, model_path='disease_model.joblib', metadata_path='model_metadata.json'):
        print(f"Loading model from {model_path}...")
        
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.symptom_columns = metadata['symptom_columns']
        self.diseases = metadata['diseases']
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        
        print("Model loaded successfully!")
    
    def predict_disease(self, symptoms_dict):
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        features = np.zeros(len(self.symptom_columns))
        
        for i, symptom in enumerate(self.symptom_columns):
            if symptom in symptoms_dict:
                features[i] = symptoms_dict[symptom]
        
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        disease_name = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_predictions.append({'disease': disease, 'probability': prob})
        
        return {
            'predicted_disease': disease_name,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(self.model.feature_importances_):
                importance_dict[self.symptom_columns[i]] = importance
            
            sorted_importance = sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)
            return sorted_importance
        else:
            return None

def main():
    print("Disease Prediction ML System")
    print("=" * 50)
    
    predictor = DiseasePredictor()
    
    dataset_path = "symptom2disease.xlsx"
    
    try:
        df = predictor.load_and_preprocess_data(dataset_path)
        if df is None:
            return
        
        X, y = predictor.prepare_features_and_labels(df)
        
        X_balanced, y_balanced = predictor.handle_class_imbalance(X, y)
        
        X_test, y_test, y_pred = predictor.train_models(X_balanced, y_balanced)
        
        predictor.save_model()
        
        importance = predictor.get_feature_importance()
        if importance:
            print(f"\nTop 10 Most Important Symptoms:")
            for symptom, imp in importance[:10]:
                print(f"{symptom:25} | Importance: {imp:.4f}")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved as 'disease_model.joblib'")
        print(f"Metadata saved as 'model_metadata.json'")
        
        print(f"\nTesting prediction...")
        sample_symptoms = {symptom: 1 for symptom in predictor.symptom_columns[:3]}
        result = predictor.predict_disease(sample_symptoms)
        
        print(f"Sample prediction:")
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
    except FileNotFoundError:
        print("Dataset file not found!")
        print("Please make sure to:")
        print("1. Download your dataset from Kaggle")
        print("2. Place it in the same directory as this script")
        print("3. Update the 'dataset_path' variable with the correct filename")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

def create_prediction_api():
    def predict_from_symptoms(selected_symptoms):
        try:
            predictor = DiseasePredictor()
            predictor.load_model()
            
            symptoms_dict = {}
            for symptom in predictor.symptom_columns:
                symptoms_dict[symptom] = 1 if symptom in selected_symptoms else 0
            
            result = predictor.predict_disease(symptoms_dict)
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    return predict_from_symptoms