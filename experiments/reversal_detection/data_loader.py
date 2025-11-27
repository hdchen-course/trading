#!/usr/bin/env python3
"""
Data Loader for Reversal Detection Experiments

Loads labeled reversal data and ensures no look-ahead bias in data splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ReversalDataLoader:
    """
    Loads and manages labeled reversal data for experiments
    
    Ensures no look-ahead bias by only using data available at each time point.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to CSV file with labeled reversal data
        """
        self.data_path = Path(data_path)
        self.df = None
        self.ground_truth = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load labeled data from CSV
        
        Returns:
            DataFrame with OHLCV, indicators, and reverse labels
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Extract ground truth labels
        if 'reverse' not in self.df.columns:
            raise ValueError("Data must contain 'reverse' column with ground truth labels")
        
        self.ground_truth = self.df['reverse'].values
        
        print(f"✅ Loaded {len(self.df)} data points")
        print(f"   Reversal points: {self.ground_truth.sum()} ({self.ground_truth.sum()/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def get_data_at_time(self, idx: int) -> pd.DataFrame:
        """
        Get data available up to a specific time point (no look-ahead)
        
        This simulates real-time conditions where only past data is available.
        
        Args:
            idx: Index of current time point
            
        Returns:
            DataFrame containing only data from times <= idx
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range [0, {len(self.df)-1}]")
        
        # Return only data up to current index
        return self.df.iloc[:idx + 1].copy()
    
    def split_time_series(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets respecting time order
        
        No shuffling to maintain temporal order and avoid look-ahead bias.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()
        
        print(f"\n📊 Time-series split:")
        print(f"   Train: {len(train_df)} points ({len(train_df)/n*100:.1f}%)")
        print(f"   Val:   {len(val_df)} points ({len(val_df)/n*100:.1f}%)")
        print(f"   Test:  {len(test_df)} points ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_features(self, exclude_labels: bool = True) -> pd.DataFrame:
        """
        Get feature columns (excluding labels and metadata)
        
        Args:
            exclude_labels: If True, exclude 'reverse' column
            
        Returns:
            DataFrame with only feature columns
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Columns to exclude
        exclude_cols = ['reverse'] if exclude_labels else []
        exclude_cols.extend(['timestamp', 'period'])  # Metadata columns
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        return self.df[feature_cols]
    
    def get_ground_truth(self) -> np.ndarray:
        """
        Get ground truth reversal labels
        
        Returns:
            Boolean array indicating reversal points
        """
        if self.ground_truth is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.ground_truth
    
    def get_available_indicators(self) -> list:
        """
        Get list of available technical indicators in the dataset
        
        Returns:
            List of indicator column names
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Exclude OHLCV, metadata, and labels
        exclude = ['timestamp', 'period', 'open', 'high', 'low', 'close', 'volume', 'reverse']
        indicators = [col for col in self.df.columns if col not in exclude]
        
        return indicators
