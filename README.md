# Ground-level NO2 Estimation using Attention-based CNN-LSTM

## Project Overview

This project addresses the **GeoAI Ground-level NO2 Estimation Challenge** from Zindi Africa, focusing on estimating ground-level nitrogen dioxide (NO2) concentrations using satellite data and machine learning techniques. NO2 is a critical air pollutant that affects human health and contributes to air quality degradation.

**Competition Link:** [Zindi GeoAI Ground-level NO2 Estimation Challenge](https://zindi.africa/competitions/geoai-ground-level-no2-estimation-challenge)

## Problem Statement

Ground-level NO2 monitoring is essential for:
- Air quality assessment and public health protection
- Environmental policy making and regulation
- Understanding urban air pollution patterns
- Climate change impact assessment

Traditional ground-based monitoring stations are expensive and sparse, making satellite-based estimation crucial for comprehensive coverage.

## Methodology

### Data Sources
The project utilizes multiple satellite datasets:
- **LST (Land Surface Temperature)**: Thermal infrared data for surface temperature
- **AAI (Aerosol Absorption Index)**: Aerosol optical depth measurements
- **Cloud Fraction**: Cloud coverage information
- **Precipitation**: Rainfall data
- **NO2 Stratospheric**: Stratospheric NO2 concentrations
- **NO2 Total**: Total atmospheric NO2
- **NO2 Tropospheric**: Ground-level NO2 concentrations
- **Tropopause Pressure**: Atmospheric pressure at tropopause level

### Data Preprocessing & Cleaning

#### Null Value Handling
- **KNN Imputation**: Missing values are filled using K-Nearest Neighbors algorithm
- **Temporal Padding**: Incomplete time sequences are padded with synthetic data
- **Feature Filtering**: Outliers and invalid measurements are removed
- **Coordinate Validation**: Latitude/longitude coordinates are verified and standardized

#### Temporal Sequence Processing
- **15-day Lookback Window**: Each prediction uses the previous 15 days of data
- **Location Grouping**: Data is organized by geographical coordinates (LAT, LON)
- **Date Sorting**: Temporal sequences are chronologically ordered
- **Sequence Padding**: Missing temporal data is filled with zero-padded sequences

### Model Architecture

The project implements an **Attention-based CNN-LSTM** architecture:

```
Input Features (8 features × 15 time steps)
           ↓
    1D Convolutional Layer
    (64 filters, kernel_size=1)
           ↓
    Bidirectional LSTM
    (64 hidden units, bidirectional)
           ↓
    Multi-head Attention
    (4 attention heads)
           ↓
    Dense Layers with BatchNorm
    (128 → 8 → 1)
           ↓
    Output: NO2 Concentration
```

#### Key Components:

1. **Convolutional Layer**: Extracts spatial-temporal features from input sequences
2. **Bidirectional LSTM**: Captures long-term dependencies in both temporal directions
3. **Multi-head Attention**: Weights important temporal patterns and features
4. **Geographical Integration**: Incorporates latitude/longitude coordinates in final layers
5. **Regularization**: Dropout (0.5) and Batch Normalization for robust training

### Training Strategy

- **Loss Function**: Root Mean Square (RMS) Loss for regression optimization
- **Optimizer**: Adam optimizer with weight decay (0.1) for regularization
- **Learning Rate**: Adaptive learning rate with ReduceLROnPlateau scheduler
- **Batch Size**: 128 samples per batch for efficient training
- **Validation**: 80-20 train-validation split with early stopping


### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Ground-level-NO2-Estimation

# Install dependencies
pip install -r requirements.txt

# Download datasets (place in repository root)
# - Train.csv
# - Test.csv
```

### Training
```bash
# Run training pipeline
python train.py

# Monitor training progress
# Check trained_models/ directory for saved checkpoints
```

### Jupyter Notebooks
- **Data Cleaning**: `data_cleaning.ipynb` - Complete data preprocessing pipeline
- **Feature Engineering**: `feature_engineering.ipynb` - Feature analysis and creation
- **Training Experiments**: `training_loop.ipynb` - Model training and evaluation

## Results & Performance

The model achieves competitive performance on the Zindi challenge. Ranking amongst the top 10 in competition leader board with a RMSE of 7.1

## Contributing

This project demonstrates advanced machine learning techniques for environmental monitoring. Contributions are welcome for:
- Model architecture improvements
- Data preprocessing enhancements
- Performance optimizations
- Documentation improvements

## License

This project is developed for the Zindi GeoAI Ground-level NO2 Estimation Challenge. Please refer to the competition terms and conditions.

## Contact

For questions about this implementation or collaboration opportunities, please refer to the project documentation and code comments.

---

**Note**: This project represents a comprehensive solution to a real-world environmental monitoring challenge, demonstrating expertise in time-series analysis, deep learning, and geospatial data processing.
