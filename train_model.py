#!/usr/bin/env python3
"""
Phishing Detection Model Trainer
================================

This script trains machine learning models for phishing detection using extracted features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class PhishingModelTrainer:
    """Train and evaluate phishing detection models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'NaiveBayes': GaussianNB()
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_importance = None
        
    def load_and_prepare_data(self, features_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the features dataset."""
        print(f"Loading features from: {features_file}")
        
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        # Load the dataset
        df = pd.read_csv(features_file)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Display basic info
        print(f"Columns: {list(df.columns)}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        
        # Prepare features and target
        # Remove non-feature columns
        exclude_cols = ['url', 'original_url', 'label', 'cse_name', 'cse_domain', 
                       'subdomain', 'domain', 'suffix', 'fqdn', 'domain_ssl_issuer']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].copy()
        X = X.fillna(-1)  # Fill missing values with -1
        
        # Convert non-numeric columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(-1)
                except:
                    # For categorical string columns, use label encoding
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
        
        # Prepare target variable
        y = self.label_encoder.fit_transform(df['label'])
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {feature_cols}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and evaluate their performance."""
        print("Training multiple models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train the model
                if model_name in ['LogisticRegression', 'SVM', 'NaiveBayes']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                if y_pred_proba is not None:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                else:
                    auc_score = None
                
                # Cross-validation
                if model_name in ['LogisticRegression', 'SVM', 'NaiveBayes']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"AUC Score: {auc_score:.4f}" if auc_score else "AUC Score: N/A")
                print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series):
        """Perform hyperparameter tuning for the best models."""
        print("Performing hyperparameter tuning...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print("Tuning Random Forest...")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        # Gradient Boosting tuning
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
        
        print("Tuning Gradient Boosting...")
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        
        # Compare tuned models
        tuned_models = {
            'Tuned_RandomForest': rf_grid.best_estimator_,
            'Tuned_GradientBoosting': gb_grid.best_estimator_
        }
        
        best_score = 0
        best_tuned_model = None
        best_tuned_name = None
        
        for name, model in tuned_models.items():
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_tuned_model = model
                best_tuned_name = name
        
        self.best_model = best_tuned_model
        self.best_model_name = best_tuned_name
        
        print(f"Best tuned model: {best_tuned_name} with accuracy: {best_score:.4f}")
        
        return tuned_models
    
    def analyze_feature_importance(self, X: pd.DataFrame):
        """Analyze feature importance for tree-based models."""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            self.feature_importance = pd.DataFrame({
                'feature': X.columns[indices],
                'importance': importances[indices]
            })
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(min(20, len(importances))), importances[indices][:20])
            plt.xticks(range(min(20, len(importances))), 
                      [X.columns[i] for i in indices[:20]], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            print("Feature importance not available for this model type")
    
    def save_model(self, filepath: str = 'phishing_detection_model.pkl'):
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_importance = model_data.get('feature_importance')
        
        print(f"Model loaded from: {filepath}")
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded")
        
        # Prepare features (same preprocessing as training)
        features_processed = features.copy()
        features_processed = features_processed.fillna(-1)
        
        # Convert to numeric
        for col in features_processed.columns:
            if features_processed[col].dtype == 'object':
                try:
                    features_processed[col] = pd.to_numeric(features_processed[col], errors='coerce')
                    features_processed[col] = features_processed[col].fillna(-1)
                except:
                    le = LabelEncoder()
                    features_processed[col] = le.fit_transform(features_processed[col].astype(str))
        
        # Scale if needed
        if self.best_model_name in ['LogisticRegression', 'SVM', 'NaiveBayes']:
            features_scaled = self.scaler.transform(features_processed)
            predictions = self.best_model.predict(features_scaled)
            probabilities = self.best_model.predict_proba(features_scaled)
        else:
            predictions = self.best_model.predict(features_processed)
            probabilities = self.best_model.predict_proba(features_processed)
        
        # Convert back to original labels
        prediction_labels = self.label_encoder.inverse_transform(predictions)
        
        return prediction_labels, probabilities
    
    def evaluate_model(self, results: Dict[str, Dict[str, Any]]):
        """Create evaluation plots and reports."""
        print("\n=== Model Evaluation Summary ===")
        
        # Create comparison plot
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        plt.bar(model_names, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # CV scores comparison
        plt.subplot(1, 2, 2)
        plt.bar(model_names, cv_means)
        plt.title('Cross-Validation Scores')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        for model_name, result in results.items():
            print(f"\n--- {model_name} ---")
            print(f"Accuracy: {result['accuracy']:.4f}")
            if result['auc_score']:
                print(f"AUC Score: {result['auc_score']:.4f}")
            print(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
            print("Classification Report:")
            print(result['classification_report'])


def main():
    """Main function to train phishing detection models."""
    # Check if features file exists
    features_file = 'phishing_features_training.csv'
    
    if not os.path.exists(features_file):
        print(f"Features file not found: {features_file}")
        print("Please run the feature extraction first:")
        print("python phishing_feature_extractor.py --batch")
        return
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    try:
        # Load and prepare data
        X, y = trainer.load_and_prepare_data(features_file)
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Evaluate models
        trainer.evaluate_model(results)
        
        # Hyperparameter tuning
        print("\nStarting hyperparameter tuning...")
        tuned_models = trainer.hyperparameter_tuning(X, y)
        
        # Analyze feature importance
        trainer.analyze_feature_importance(X)
        
        # Save the best model
        trainer.save_model()
        
        print("\nTraining completed successfully!")
        print(f"Best model: {trainer.best_model_name}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
