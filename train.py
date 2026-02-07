# train.py
"""
Nigerian House Rent Price Prediction - Model Training
This script trains a regression model using PyCaret and saves it for deployment
"""

import pandas as pd
import os
from pycaret.regression import *

def train_model():
    """
    Train regression model using PyCaret
    """
    
    print("=" * 60)
    print(" NIGERIAN HOUSE RENT PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n Step 1: Loading dataset...")
    
    # Load the dataset
    data = pd.read_csv('data/house_rent.csv')
    
    print(f" Dataset loaded successfully!")
    print(f"   - Total samples: {len(data)}")
    print(f"   - Features: {len(data.columns) - 1}")
    print(f"   - Target: monthly_rent")
    
    print("\n First few rows:")
    print(data.head())
    
    # ==================== STEP 2: SETUP PYCARET ====================
    print("\n Step 2: Setting up PyCaret environment...")
    print("   (This may take a minute...)")
    
    # Initialize PyCaret setup
    # target = 'monthly_rent' (what we want to predict)
    # session_id = 42 (for reproducibility)
    # train_size = 0.8 (80% training, 20% testing)
    # categorical_features = list of text columns
    # numeric_features = list of number columns
    
    reg_setup = setup(
        data=data,
        target='monthly_rent',
        session_id=42,
        train_size=0.8,
        fold=5,  # 5-fold cross-validation
        normalize=True,  # Normalize numeric features
        transformation=False,
        verbose=False  # Reduce output
    )
    
    print(" Setup complete!")
    
    # ==================== STEP 3: COMPARE MODELS ====================
    print("\n Step 3: Comparing different regression models...")
    print("   (Testing 10+ algorithms to find the best one...)")
    
    # Compare all available models
    # PyCaret will train multiple models and rank them by performance
    best_model = compare_models(
        n_select=1,  # Select the best model
        sort='MAE'   # Sort by Mean Absolute Error (lower is better)
    )
    
    print(f"\n Best model selected: {type(best_model).__name__}")
    
    # ==================== STEP 4: FINALIZE MODEL ====================
    print("\n Step 4: Finalizing model on entire dataset...")
    
    # Train the best model on the entire dataset (not just training set)
    final_model = finalize_model(best_model)
    
    print(" Model finalized!")
    
    # ==================== STEP 5: SAVE MODEL ====================
    print("\n Step 5: Saving model for deployment...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the final model
    save_model(final_model, 'models/rent_model')
    
    print(" Model saved to: models/rent_model.pkl")
    
    # ==================== STEP 6: MODEL PERFORMANCE ====================
    print("\n Step 6: Model Performance Summary")
    print("=" * 60)
    
    # Get predictions on test set
    predictions = predict_model(final_model)
    
    # Calculate and display metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(predictions['monthly_rent'], predictions['prediction_label'])
    r2 = r2_score(predictions['monthly_rent'], predictions['prediction_label'])
    
    print(f"\n Mean Absolute Error (MAE): ₦{mae:,.0f}")
    print(f"   → On average, predictions are off by ₦{mae:,.0f}")
    
    print(f"\n R² Score: {r2:.4f}")
    print(f"   → Model explains {r2*100:.2f}% of rent price variation")
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE!")
    print("=" * 60)
    print("\n Next steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Test predictions with different inputs")
    print("   3. Deploy to Streamlit Cloud")
    print("\n")

if __name__ == "__main__":
    train_model()