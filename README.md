# 📈 Stock Market Directional Movement Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Learning-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)

> **Advanced time-series forecasting solution for predicting stock price directional movements using ensemble machine learning techniques**

---

## 🎯 Competition Overview

This repository contains our solution for the **Kaggle Stock Market Prediction Competition**, where participants predict the directional movement (up or down) of stock prices based on historical trading data.

### Problem Statement

Given historical trading data for 94 stocks across 500 trading days, predict the **probability of upward price movement** from the opening to closing price on day 10, using:
- Opening, closing, maximum, and minimum prices for days 1-9
- Trading volume for days 1-9
- Opening price for day 10

**Target Output:** A probability score in the range [0, 1], where:
- `1.0` = Stock will definitely move up
- `0.0` = Stock will definitely move down

---

## 🏆 Key Achievements

- ✅ Implemented ensemble learning approach combining Ridge Regression and Random Forest
- ✅ Developed advanced feature engineering pipeline for time-series data
- ✅ Achieved robust predictions through model stacking techniques
- ✅ Handled normalized price data and non-consecutive trading days
- ✅ Created reproducible, well-documented codebase

---

## 📊 Dataset Characteristics

### Training Data (`training.csv`)
- **94 stocks** with 500 consecutive trading days each
- **Features per day:**
  - Opening price (normalized to first day)
  - Maximum price
  - Minimum price
  - Closing price
  - Trading volume
- **Format:** Each row represents one stock's complete time series

### Test Data (`test.csv`)
- **25 time segments** × **94 stocks** = 2,350 predictions
- **9 days** of historical data + day 10 opening price
- Randomly sampled from the year following training period (no overlap)
- Non-consecutive trading days (excludes market closures/holidays)

### Data Preprocessing
- All price data normalized to first day opening price
- Handles market holidays and non-trading days automatically
- Time-series validation to prevent data leakage

---

## 🛠️ Technical Architecture

### Model Pipeline

```
Raw Data → Feature Engineering → Model Training → Model Stacking → Predictions
```

### Core Components

#### 1. **Feature Engineering**
- Temporal features (day-to-day price changes)
- Statistical aggregations (moving averages, volatility)
- Volume-price relationships
- Momentum indicators

#### 2. **Base Models**

**Ridge Regression**
- Linear model with L2 regularization
- Captures linear relationships in price movements
- Fast training and prediction
- Hyperparameter tuning via grid search

**Random Forest Regression**
- Non-linear ensemble of decision trees
- Captures complex feature interactions
- Robust to outliers and noise
- Feature importance analysis

#### 3. **Model Stacking**
- Meta-learner combining predictions from both base models
- Cross-validation to prevent overfitting
- Weighted ensemble for optimal performance

---

## 📁 Repository Structure

```
Stock-market-prediction-/
│
├── model_tuner.py              # Hyperparameter optimization
├── model_tuner.ipynb           # Interactive tuning notebook
├── model_stacker.py            # Ensemble stacking implementation
├── model_stacker.ipynb         # Stacking analysis notebook
│
├── training.csv                # Official training dataset
├── test.csv                    # Official test dataset
│
└── predictions/                # Submission files
    └── stacker/                # Winning submission (ensemble)
        └── final_predictions.csv
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
jupyter >= 1.0.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Valgha/Stock-market-prediction-.git
cd Stock-market-prediction-

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Hyperparameter Tuning

```bash
# Run parameter optimization
python model_tuner.py

# Or use the interactive notebook
jupyter notebook model_tuner.ipynb
```

#### 2. Generate Predictions

```bash
# Run the stacked ensemble model
python model_stacker.py

# Predictions will be saved to predictions/stacker/
```

#### 3. Explore Analysis Notebooks

```bash
jupyter notebook model_stacker.ipynb
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Analyzed price movement patterns across 94 stocks
- Investigated volume-price correlations
- Identified temporal patterns and market regimes

### 2. Feature Engineering
Created derived features including:
- Price momentum indicators
- Volatility measures
- Volume-weighted metrics
- Relative strength indicators
- Moving average convergence/divergence

### 3. Model Selection & Tuning
- Evaluated multiple regression algorithms
- Grid search for optimal hyperparameters
- Cross-validation on time-series data
- Selected Ridge and Random Forest as complementary models

### 4. Ensemble Learning
- Stacked predictions using meta-learning
- Optimized weights for base model combination
- Validated on held-out time segments

---

## 📈 Model Performance

The ensemble approach leverages the strengths of both models:
- **Ridge Regression:** Stable linear baseline with low variance
- **Random Forest:** Captures non-linear patterns and interactions
- **Stacked Ensemble:** Achieves superior performance through intelligent combination

*Note: Detailed performance metrics available in the competition leaderboard*

---

## 👥 Team

**Team Members:**
- [Vishal Khatawate](https://github.com/Valgha)
- Sachin Khatavate

---

## 📚 Key Learnings

1. **Time-series data requires special handling** to prevent look-ahead bias
2. **Feature engineering is crucial** for stock prediction tasks
3. **Ensemble methods** consistently outperform individual models
4. **Market holidays and gaps** must be properly accounted for
5. **Normalized data** enables better model generalization

---

## 🔮 Future Improvements

- [ ] Incorporate sentiment analysis from financial news
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Experiment with LSTM/GRU neural networks
- [ ] Implement attention mechanisms for temporal patterns
- [ ] Add external market indicators (VIX, sector indices)
- [ ] Deploy as real-time prediction API

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

Vishal Khatawate - [@Valgha](https://github.com/Valgha)

Project Link: [https://github.com/Valgha/Stock-market-prediction-](https://github.com/Valgha/Stock-market-prediction-)

---

## 🙏 Acknowledgments

- Kaggle for hosting the competition and providing the dataset
- The open-source community for excellent machine learning libraries
- Fellow competitors for inspiring approaches and discussions

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>
