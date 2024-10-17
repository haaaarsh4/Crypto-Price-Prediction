

# Bitcoin Price Prediction with Sentiment Analysis

This project focuses on predicting the price of Bitcoin (BTC) using historical market data along with sentiment analysis of Wikipedia edits related to Bitcoin. By combining time-series analysis with natural language processing (NLP), the project aims to explore how market trends and public sentiment influence cryptocurrency prices.

## Overview

The core components of this project involve:

1. **Historical Bitcoin Data**: We gather historical price data for Bitcoin from Yahoo Finance and use this data to predict future prices.
   
2. **Sentiment Analysis of Wikipedia Edits**: We scrape the revision history of the Bitcoin Wikipedia page, analyze the sentiment of user comments on these edits, and incorporate this into our analysis.

3. **LSTM Model for Price Prediction**: A Long Short-Term Memory (LSTM) neural network is trained on historical price data to forecast future prices. This model is commonly used in time-series forecasting tasks due to its ability to capture temporal dependencies.

4. **Data Visualization**: The actual vs predicted prices are plotted to evaluate the model's performance.

## Dependencies

To run the project, you'll need to install the following Python libraries:

- **NumPy**: Efficient array manipulation.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib**: Plotting and data visualization.
- **Yahoo Finance (`yfinance`)**: To fetch Bitcoin price data.
- **TensorFlow / Keras**: To build and train the LSTM neural network.
- **MWClient**: To interact with Wikipedia's API and fetch revision history.
- **Transformers**: Hugging Face's library for sentiment analysis using pre-trained models.

## How It Works

### 1. Data Collection

- **Bitcoin Price Data**: We use Yahoo Finance API (`yfinance`) to collect historical BTC price data in USD.
  
- **Sentiment Data**: Using the Wikipedia API via `mwclient`, we retrieve the revision history of the Bitcoin page. The comments associated with each revision are analyzed for sentiment using Hugging Face's `transformers` library. The sentiment is classified as either positive or negative, and a sentiment score is computed for each comment.

### 2. Data Preprocessing

- **Market Data**: The closing prices of Bitcoin are scaled between 0 and 1 using `MinMaxScaler`, which is necessary for training the LSTM model.
  
- **Sentiment Data**: We aggregate the sentiment scores by date and compute the average sentiment and the proportion of negative sentiments.

- **Training Data**: We use a sliding window approach to prepare the data for the LSTM model. The last 60 days of historical prices are used to predict the price on the 61st day.

### 3. Model Training

We train a three-layer LSTM neural network:
- **First Layer**: LSTM with 50 units, returning sequences.
- **Second Layer**: LSTM with 50 units.
- **Third Layer**: Dense layer to output a single price prediction.

Each LSTM layer is followed by a Dropout layer to prevent overfitting. The model is trained using `mean_squared_error` loss and the Adam optimizer.

### 4. Testing and Prediction

- We test the model on unseen data, predicting Bitcoin prices for a specific date range (January to April 2024, for instance).
  
- The predicted prices are compared against the actual market prices, and both are plotted to visually evaluate the performance of the model.

### 5. Visualization

The project generates a plot comparing the actual and predicted Bitcoin prices over time. This helps in visually assessing how well the model performed in forecasting the price movement.

## Results

The trained LSTM model outputs predictions for the future price of Bitcoin, which are visualized alongside the actual price data. The impact of Wikipedia sentiment analysis on the predictions could be further explored by incorporating this into the feature set of the model.

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-price-prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib yfinance mwclient transformers tensorflow
   ```

3. Run the script:
   ```bash
   python bitcoin_price_prediction.py
   ```

4. Review the output graphs showing the actual and predicted Bitcoin prices.

## Future Enhancements

Some ideas for future work include:
- **Incorporating More Sentiment Sources**: Analyzing other sources of sentiment, such as Twitter, Reddit, or financial news articles.
- **Feature Engineering**: Incorporating more financial indicators (e.g., volume, moving averages) or macroeconomic data.
- **Fine-tuning the LSTM Model**: Experimenting with different hyperparameters, network architectures, or optimizers.
- **Predicting Price Volatility**: Expanding the model to predict not just the price but also price volatility, which is crucial in cryptocurrency trading.

## Conclusion

This project demonstrates how time-series forecasting techniques like LSTM can be used to predict cryptocurrency prices. By integrating sentiment analysis of Wikipedia revisions, we explore the potential impact of public sentiment on Bitcoin price movements.
