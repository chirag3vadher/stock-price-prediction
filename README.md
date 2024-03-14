# Stock Price Prediction Project

This project aims to develop a machine learning model to predict stock prices using historical data and financial indicators.

## Project Overview

The project involves the following key steps:

1. **Data Collection**: Fetch historical stock price data using the `yfinance` library.
2. **Feature Engineering**: Calculate financial indicators such as moving averages, RSI (Relative Strength Index), etc., from the historical data.
3. **Data Preprocessing**: Clean the data, handle missing values, and normalize the data.
4. **Model Building**: Train machine learning models such as Linear Regression, Random Forest, or LSTM on the training data.
5. **Model Evaluation**: Evaluate the models on the validation set using metrics like MSE or MAPE.
6. **Hyperparameter Tuning**: Fine-tune the model parameters to improve performance.
7. **Final Evaluation**: Test the final model on the test set to assess its performance.
8. **Deployment**: Deploy the model for predictions on new stock data.

## Project Structure

The project structure is as follows:

stock_price_prediction

- |-- data/ 
  - | |-- raw/ # Raw historical stock price data
  - | |-- processed/ # Processed data with financial indicators
- |-- models/ # Saved models and related files
- |-- notebooks/ # Jupyter notebooks for data exploration and analysis
- |-- src/ # Source code for the project
  - | |-- init.py
  - | |-- data_collection.py # Code to fetch historical stock data
   - | |-- feature_engineering.py # Code to compute financial indicators
    - | |-- data_preprocessing.py # Code to preprocess the data
    - | |-- model_building.py # Code to build and train models
    - | |-- model_evaluation.py # Code to evaluate model performance
    - | |-- utils.py # Utility functions
- |-- tests/ # Unit tests for the project modules
    - | |-- init.py
   -  | |-- test_data_collection.py
   -  | |-- test_feature_engineering.py
   -  | |-- test_data_preprocessing.py
    - | |-- test_model_building.py
    - | |-- test_model_evaluation.py
- |-- requirements.txt # Project dependencies

## Usage

To run the project, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Execute the `main.py` script located in the `src/` directory to run the entire pipeline.

## Detailed Explanation

### Data Collection
- We chose the `yfinance` library for fetching historical stock price data due to its ease of use and extensive coverage of stock data.

### Feature Engineering
- We calculated financial indicators like following
  - Trend Volatility Ratio (TVR)
  - Momentum Strength Index (MSI)
  - Adaptive Trend Indicator (ATI)
  - Volume Strength Ratio (VSR)
  - Dynamic Volatility Index (DVI)
  - Price Momentum Oscillator (PMO)
  - Adaptive Volume Efficiency (AVE)
  - Trend Momentum Quotient (TMQ)
  - Dynamic Price Action (DPA)
  - Market Sentiment Index (MSI)
  - Price Trend Magnitude (PTM)
  - Volume Efficiency Ratio (VER)
  - Price Trend Strength (PTS)
  - Adaptive Momentum Indicator (AMI)
  - Volume Price Ratio (VPR)
### Data Preprocessing
- Data preprocessing involved cleaning the data, handling missing values, and normalizing the data to ensure that it is suitable for training machine learning models.

### Model Building
- We experimented with two  machine learning models including Linear Regression and random forest to determine the best approach for stock price prediction.

### Model Evaluation
- Model evaluation was performed using metrics like MSE and MAPE on a validation set to assess the performance of the trained models.

### Hyperparameter Tuning
- Hyperparameter tuning was conducted to fine-tune the model parameters and improve performance using techniques like Grid Search.

### Final Evaluation
- The final model was evaluated on a test set to provide an unbiased estimate of its performance on unseen data.

### Deployment
- Once the model was trained and evaluated, it was deployed for predictions on new stock data in real-time using the Flask Api framework.

## Results
- The performance of the trained models is evaluated using appropriate metric (MSE), and the best-performing model is selected for deployment.

## Conclusion
The Stock Price Prediction project demonstrates the application of machine learning techniques to forecast stock prices. 
By following established standards and best practices, the project aims to deliver accurate predictions and provide valuable insights for investors and traders in the financial markets.


## Stretch Objectives and future scope
### 1. Real-Time Data Handling and Live Predictions
#### Approach:
- **Data Streaming Pipeline:**
  - Implement a streaming pipeline using Kafka or AWS Kinesis to ingest real-time stock market data.
  - Ensure data is continuously collected from reliable sources and stored in a scalable data store like S3
- **Real-Time Preprocessing:**
  - Develop a preprocessing module that can handle incoming data streams in real-time.
  - Apply necessary transformations and feature extraction techniques to prepare data for model inference.
- **Live Predictions:**
  - Implement a microservice architecture using Flask or FastAPI to serve live predictions.
  - Configure the application to trigger predictions upon receiving new data events using services like event bridge.

#### Pseudo-Code:
```python
# Pseudo-code for Real-Time Predictions
while True:
    new_data = stream_data()  # Fetch new data from data stream
    preprocessed_data = preprocess_real_time(new_data)  # Preprocess incoming data
    prediction = make_live_prediction(model, preprocessed_data)  # Make prediction
    publish_prediction(prediction)  # Publish prediction to API or dashboard
```

### 2. Optimal Portfolio Recommendation

#### Approach:
- **Expected Returns and Risk Prediction:**
  - Enhance the model to predict not only stock prices but also expected returns and risks.
  - Incorporate financial metrics such as beta to find volatility
- **Portfolio Optimization:**
  - Implement portfolio optimization algorithms to construct optimal portfolios.
  - Consider investor preferences, risk tolerance, and constraints while optimizing portfolios.

#### Pseudo-Code:
```python
# Pseudo-code for Portfolio Optimization
returns, risks = predict_returns_risks(stocks)  # Predict returns and risks for each stock
optimal_portfolio = optimize_portfolio(returns, risks, constraints)  # Optimize portfolio allocation
```

### 3. Market Sentiment Analysis Integration

#### Approach:
- **Textual Data Collection:**
  - Collect textual data from news articles, social media platforms, and other sources relevant to the stock market.
- **Sentiment Analysis:**
  - Apply natural language processing (NLP) techniques to analyze sentiment in textual data.
  - Utilize pre-trained sentiment analysis models or train custom models on financial text data.
- **Feature Engineering:**
  - Integrate sentiment scores as additional features into the predictive model.

#### Pseudo-Code:
```python
# Pseudo-code for Sentiment Analysis Integration
news_data = fetch_news_data()  # Fetch news articles related to stocks
sentiment_scores = analyze_sentiment(news_data)  # Analyze sentiment in news articles
features = combine_features(financial_indicators, sentiment_scores)  # Combine with financial features
model_input = preprocess_data(features)  # Preprocess data for model input
prediction = make_prediction(model, model_input)  # Make prediction using enhanced model 
```