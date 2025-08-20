#!/usr/bin/env python3
"""
Script to explore and understand the phishing detection dataset structure.
"""

import pandas as pd
import os
import sys

def explore_training_dataset():
    """Explore the training dataset to understand its structure."""
    
    # Path to the training dataset
    training_file = "/home/vk/phishing/phishing/PS02_Training_set/PS02_Training_set/PS02_Training_set.xlsx"
    
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        return None
    
    try:
        # Read the Excel file
        print("Reading training dataset...")
        df = pd.read_excel(training_file)
        
        print(f"\n=== Dataset Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\n=== First 5 rows ===")
        print(df.head())
        
        print(f"\n=== Data types ===")
        print(df.dtypes)
        
        print(f"\n=== Missing values ===")
        print(df.isnull().sum())
        
        # Look for label column (phishing/legitimate)
        print(f"\n=== Unique values in each column ===")
        for col in df.columns:
            unique_vals = df[col].unique()
            print(f"{col}: {len(unique_vals)} unique values")
            if len(unique_vals) < 10:
                print(f"  Values: {unique_vals}")
            
        return df
        
    except Exception as e:
        print(f"Error reading training file: {e}")
        return None

def explore_domains_dataset():
    """Explore the domains dataset."""
    
    # Path to the domains dataset
    domains_file = "/home/vk/phishing/phishing/PS-02 Phishing Detection CSE_Domains_Dataset_for_Stage_1.xlsx"
    
    if not os.path.exists(domains_file):
        print(f"Domains file not found: {domains_file}")
        return None
    
    try:
        # Read the Excel file
        print("\n\n=== DOMAINS DATASET ===")
        print("Reading domains dataset...")
        df = pd.read_excel(domains_file)
        
        print(f"\n=== Dataset Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\n=== First 5 rows ===")
        print(df.head())
        
        print(f"\n=== Data types ===")
        print(df.dtypes)
        
        print(f"\n=== Missing values ===")
        print(df.isnull().sum())
        
        # Look for label column (phishing/legitimate)
        print(f"\n=== Unique values in each column ===")
        for col in df.columns:
            unique_vals = df[col].unique()
            print(f"{col}: {len(unique_vals)} unique values")
            if len(unique_vals) < 10:
                print(f"  Values: {unique_vals}")
            
        return df
        
    except Exception as e:
        print(f"Error reading domains file: {e}")
        return None

if __name__ == "__main__":
    print("Exploring phishing detection datasets...")
    
    # Explore training dataset
    training_df = explore_training_dataset()
    
    # Explore domains dataset  
    domains_df = explore_domains_dataset()
    
    print("\n=== Dataset Exploration Complete ===")
