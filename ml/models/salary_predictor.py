"""
Salary Prediction Model using TensorFlow
Predicts salary ranges for AI/ML positions based on job features
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalaryPredictor:
    """Deep learning model for salary prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def extract_features(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from job data"""
        logger.info("Extracting features from job data...")
        
        features = pd.DataFrame()
        
        # 1. Seniority level encoding
        if 'seniority_level' not in self.label_encoders:
            self.label_encoders['seniority_level'] = LabelEncoder()
            seniority_encoded = self.label_encoders['seniority_level'].fit_transform(
                jobs_df['seniority_level'].fillna('Unknown')
            )
        else:
            # Handle unseen labels - map to closest known label
            seniority_values = jobs_df['seniority_level'].fillna('Mid-level')
            known_levels = set(self.label_encoders['seniority_level'].classes_)
            
            # Map unknown values to closest known ones
            seniority_mapping = {
                'Principal': 'Senior' if 'Senior' in known_levels else 'Mid-level',
                'Lead': 'Senior' if 'Senior' in known_levels else 'Mid-level',
                'Staff': 'Senior' if 'Senior' in known_levels else 'Mid-level',
                'Unknown': 'Mid-level'
            }
            
            seniority_values = seniority_values.apply(
                lambda x: x if x in known_levels else seniority_mapping.get(x, 'Mid-level')
            )
            seniority_encoded = self.label_encoders['seniority_level'].transform(seniority_values)
        features['seniority_encoded'] = seniority_encoded
        
        # 2. Remote work indicator
        features['is_remote'] = jobs_df['remote'].astype(int)
        
        # 3. Source platform encoding
        if 'source' not in self.label_encoders:
            self.label_encoders['source'] = LabelEncoder()
            source_encoded = self.label_encoders['source'].fit_transform(
                jobs_df['source'].fillna('Unknown')
            )
        else:
            # Handle unseen labels - use the first known source as default
            source_values = jobs_df['source'].fillna('RemoteOK')
            known_sources = set(self.label_encoders['source'].classes_)
            default_source = list(known_sources)[0] if known_sources else 'RemoteOK'
            source_values = source_values.apply(lambda x: x if x in known_sources else default_source)
            source_encoded = self.label_encoders['source'].transform(source_values)
        features['source_encoded'] = source_encoded
        
        # 4. Job type encoding
        if 'job_type' not in self.label_encoders:
            self.label_encoders['job_type'] = LabelEncoder()
            job_type_encoded = self.label_encoders['job_type'].fit_transform(
                jobs_df['job_type'].fillna('Full-time')
            )
        else:
            # Handle unseen labels
            job_type_values = jobs_df['job_type'].fillna('Full-time')
            known_types = set(self.label_encoders['job_type'].classes_)
            job_type_values = job_type_values.apply(lambda x: x if x in known_types else 'Full-time')
            job_type_encoded = self.label_encoders['job_type'].transform(job_type_values)
        features['job_type_encoded'] = job_type_encoded
        
        # 5. Number of requirements
        features['num_requirements'] = jobs_df['requirements'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # 6. Key skill indicators
        key_skills = [
            'Python', 'TensorFlow', 'PyTorch', 'Machine Learning',
            'Deep Learning', 'SQL', 'Docker', 'Kubernetes', 'AWS',
            'JavaScript', 'React', 'Node.js', 'Java', 'Scala'
        ]
        
        for skill in key_skills:
            features[f'has_{skill.lower().replace(" ", "_")}'] = jobs_df['requirements'].apply(
                lambda reqs: 1 if isinstance(reqs, list) and skill in str(reqs) else 0
            )
        
        # 7. Title features
        title_keywords = ['Senior', 'Lead', 'Principal', 'Staff', 'Junior', 
                         'Manager', 'Director', 'VP', 'Engineer', 'Scientist',
                         'Developer', 'Architect', 'Analyst']
        
        for keyword in title_keywords:
            features[f'title_has_{keyword.lower()}'] = jobs_df['title'].apply(
                lambda x: 1 if keyword.lower() in str(x).lower() else 0
            )
        
        # 8. Description length (proxy for detail/complexity)
        features['description_length'] = jobs_df['description'].apply(
            lambda x: len(str(x)) if x else 0
        )
        
        # 9. TF-IDF features from description (reduced dimensionality)
        if not hasattr(self, 'tfidf_fitted'):
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                jobs_df['description'].fillna('')
            )
            self.tfidf_fitted = True
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(
                jobs_df['description'].fillna('')
            )
        
        # Add top TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        features = pd.concat([features, tfidf_df], axis=1)
        
        # 10. Company features (if we had more data, we'd encode company reputation)
        features['company_name_length'] = jobs_df['company'].apply(
            lambda x: len(str(x)) if x else 0
        )
        
        logger.info(f"Extracted {features.shape[1]} features")
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def prepare_targets(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Prepare target variables (salary ranges)"""
        # For now, predict the midpoint of the salary range
        # In production, you might predict both min and max
        
        salary_min = jobs_df['salary_min'].fillna(50000)  # Default minimum
        salary_max = jobs_df['salary_max'].fillna(150000)  # Default maximum
        
        # Use log transformation for better distribution
        salary_midpoint = (salary_min + salary_max) / 2
        salary_log = np.log1p(salary_midpoint)
        
        return salary_log
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture"""
        logger.info(f"Building model with input dimension: {input_dim}")
        
        model = models.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer (regression)
            layers.Dense(1, activation='linear')
        ])
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, jobs_file: str, epochs: int = 100, batch_size: int = 32):
        """Train the salary prediction model"""
        logger.info(f"Loading training data from {jobs_file}")
        
        # Load data
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)
        
        jobs_df = pd.DataFrame(jobs_data)
        
        # Filter out jobs without salary data for training
        jobs_with_salary = jobs_df[
            (jobs_df['salary_min'].notna()) & 
            (jobs_df['salary_max'].notna()) &
            (jobs_df['salary_min'] > 0) &
            (jobs_df['salary_max'] > 0)
        ].copy()
        
        if len(jobs_with_salary) < 10:
            logger.warning(f"Only {len(jobs_with_salary)} jobs with salary data. Using synthetic data.")
            # Generate synthetic training data for demonstration
            jobs_with_salary = self.generate_synthetic_data(1000)
        
        logger.info(f"Training on {len(jobs_with_salary)} samples")
        
        # Extract features and targets
        X = self.extract_features(jobs_with_salary)
        y = self.prepare_targets(jobs_with_salary)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate
        val_loss, val_mae, val_mse = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation MAE: ${np.expm1(val_mae):,.0f}")
        logger.info(f"Validation RMSE: ${np.sqrt(np.expm1(val_mse)):,.0f}")
        
        # Save model
        self.save_model()
        
        return history
    
    def generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        
        seniority_levels = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Principal']
        sources = ['LinkedIn', 'Indeed', 'RemoteOK']
        job_types = ['Full-time', 'Contract', 'Part-time']
        
        data = []
        for i in range(n_samples):
            seniority = np.random.choice(seniority_levels)
            
            # Base salary by seniority
            base_salaries = {
                'Junior': 70000,
                'Mid-level': 100000,
                'Senior': 140000,
                'Lead': 170000,
                'Principal': 200000
            }
            
            base = base_salaries[seniority]
            salary_min = base + np.random.normal(0, 10000)
            salary_max = salary_min + np.random.uniform(20000, 50000)
            
            # Generate requirements
            all_skills = ['Python', 'TensorFlow', 'PyTorch', 'SQL', 'Docker', 
                         'Kubernetes', 'AWS', 'Machine Learning', 'Deep Learning']
            num_skills = np.random.randint(3, 8)
            requirements = np.random.choice(all_skills, num_skills, replace=False).tolist()
            
            data.append({
                'title': f"{seniority} ML Engineer",
                'company': f"Company_{i}",
                'description': f"Job description for {seniority} position requiring {', '.join(requirements)}",
                'requirements': requirements,
                'salary_min': max(30000, salary_min),
                'salary_max': min(500000, salary_max),
                'seniority_level': seniority,
                'remote': np.random.choice([True, False]),
                'source': np.random.choice(sources),
                'job_type': np.random.choice(job_types)
            })
        
        return pd.DataFrame(data)
    
    def predict(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Predict salaries for new jobs"""
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return None
        
        # Extract features
        X = self.extract_features(jobs_df)
        
        # Ensure same features as training
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.feature_columns]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict (returns log-transformed values)
        predictions_log = self.model.predict(X_scaled, verbose=0)
        
        # Convert back from log scale
        predictions = np.expm1(predictions_log).flatten()
        
        return predictions
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('ml/models/training_history.png')
        logger.info("Training history saved to ml/models/training_history.png")
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        # Save TensorFlow model
        self.model.save('ml/models/salary_model.h5')
        
        # Save preprocessors
        with open('ml/models/preprocessors.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }, f)
        
        logger.info("Model saved to ml/models/")
    
    def load_model(self):
        """Load a trained model"""
        try:
            # Load TensorFlow model
            self.model = keras.models.load_model('ml/models/salary_model.h5')
            
            # Load preprocessors
            with open('ml/models/preprocessors.pkl', 'rb') as f:
                preprocessors = pickle.load(f)
                self.scaler = preprocessors['scaler']
                self.tfidf_vectorizer = preprocessors['tfidf_vectorizer']
                self.label_encoders = preprocessors['label_encoders']
                self.feature_columns = preprocessors['feature_columns']
            
            self.is_trained = True
            self.tfidf_fitted = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def explain_prediction(self, job: Dict) -> Dict:
        """Explain salary prediction for a specific job"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        job_df = pd.DataFrame([job])
        
        # Get prediction
        predicted_salary = self.predict(job_df)[0]
        
        # Extract features for explanation
        features = self.extract_features(job_df)
        
        # Key factors (simplified - in production use SHAP or LIME)
        factors = {
            'seniority_level': job.get('seniority_level', 'Unknown'),
            'is_remote': job.get('remote', False),
            'num_requirements': len(job.get('requirements', [])),
            'has_ml_skills': any(skill in str(job.get('requirements', [])) 
                                for skill in ['Machine Learning', 'Deep Learning']),
            'title_has_senior': 'senior' in job.get('title', '').lower()
        }
        
        return {
            'predicted_salary': predicted_salary,
            'confidence_interval': (predicted_salary * 0.85, predicted_salary * 1.15),
            'key_factors': factors,
            'market_position': 'Above Average' if predicted_salary > 120000 else 'Average'
        }


if __name__ == "__main__":
    import os
    
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Get data path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    jobs_file = os.path.join(base_dir, 'ml', 'data', 'jobs_data.json')
    
    # Train model
    logger.info("Training salary prediction model...")
    history = predictor.train(jobs_file, epochs=50)
    
    # Test predictions
    test_job = {
        'title': 'Senior Machine Learning Engineer',
        'company': 'Tech Corp',
        'description': 'Looking for an experienced ML engineer with deep learning expertise',
        'requirements': ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'AWS'],
        'seniority_level': 'Senior',
        'remote': True,
        'source': 'LinkedIn',
        'job_type': 'Full-time'
    }
    
    test_df = pd.DataFrame([test_job])
    prediction = predictor.predict(test_df)
    
    print(f"\n📊 Salary Prediction for: {test_job['title']}")
    print(f"💰 Predicted Salary: ${prediction[0]:,.0f}")
    
    # Explain prediction
    explanation = predictor.explain_prediction(test_job)
    print(f"📈 Confidence Range: ${explanation['confidence_interval'][0]:,.0f} - ${explanation['confidence_interval'][1]:,.0f}")
    print(f"🎯 Market Position: {explanation['market_position']}")