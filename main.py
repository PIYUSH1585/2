#9th Try

import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load extended dataset
data = pd.read_csv('ForEx Rates.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Reverse the order of dates
data = data.sort_index(ascending=False)

# Function to train polynomial regression model for a given currency
def train_model(currency):
    # Filter data for the desired currency
    currency_data = data[[currency]].dropna()

    # Prepare features (X: dates as ordinal values, y: exchange rates)
    X = currency_data.index.map(datetime.toordinal).values.reshape(-1, 1)
    y = currency_data[currency].values

    # Create polynomial features
    degree = 2  # Adjust the degree as needed
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Train polynomial regression model
    model.fit(X, y)

    return model

# Function to plot historical exchange rates
def plot_historical_rates(currency_data, currency_code):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(currency_data.index, currency_data[currency_code], marker='o', linestyle='-')
    ax.set_title(f'Historical Exchange Rates for {currency_code}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate in Rs ')
    ax.grid(True)
    st.pyplot(fig)

# Dictionary mapping currency codes to country names
currency_countries = {
    "USD": "US Dollar",
    "AUS": "Australian Dollar",
    "CAN": "Canadian Dollar",
    "SWF": "Swiss Franc",
    "DKK": "Danish Krone",
    "EUR": "Euro",
    "GBP": "British Pound",
    "HKD": "Hong Kong Dollar",
    "IDR": "Indonesian Rupiah",
    "JPY": "Japanese Yen",
    "KRW": "South Korean Won",
    "MYR": "Malaysian Ringgit",
    "NOK": "Norwegian Krone",
    "NZD": "New Zealand Dollar",
    "SEK": "Swedish Krona",
    "SGD": "Singapore Dollar",
    "THB": "Thai Baht",
    "TWD": "New Taiwan Dollar",
    "ZAR": "South African Rand",
    "AED": "United Arab Emirates Dirham",
    "ARS": "Argentine Peso",
    "BRL": "Brazilian Real",
    "CNY": "Chinese Yuan",
    "HUF": "Hungarian Forint",
    "ILS": "Israeli New Shekel",
    "ISK": "Icelandic Króna",
    "MXN": "Mexican Peso",
    "PHP": "Philippine Peso",
    "PKR": "Pakistani Rupee",
    "PLN": "Polish Złoty",
    "RUB": "Russian Ruble",
    "SAR": "Saudi Riyal",
    "TRY": "Turkish Lira",
    "VEF": "Venezuelan Bolívar",
    "BHD": "Bahraini Dinar"
}

# Streamlit App
st.title('Currency Exchange Rate Prediction')

# Sidebar inputs
selected_currency = st.sidebar.selectbox('Select Currency', list(currency_countries.keys()))
selected_currency_code = selected_currency  # Extract currency code from selection
start_date = st.sidebar.date_input('Start Date', datetime(2024, 5, 1))
end_date = st.sidebar.date_input('End Date', datetime(2024, 12, 31))
show_historical_chart = st.sidebar.checkbox('Show Historical Exchange Rate Chart')

# Apply button
if st.sidebar.button('Apply'):
    # Convert start_date to datetime object
    start_date = datetime.combine(start_date, datetime.min.time())

    # Check if start date is before 2023
    if start_date.date() < datetime(2023, 1, 1).date():
        st.warning("Start date is before 2023. Displaying historical data.")
        historical_data = data.loc[:datetime(2022, 12, 31), [selected_currency_code]]
        st.write(historical_data)
        if show_historical_chart:
            plot_historical_rates(historical_data, selected_currency_code)
    else:
        # Train model for selected currency
        model = train_model(selected_currency_code)

        # Predict exchange rates for the selected period
        prediction_dates = pd.date_range(start_date, end_date)
        prediction_ordinals = prediction_dates.map(datetime.toordinal).values.reshape(-1, 1)
        predictions = model.predict(prediction_ordinals)

        # Display predictions
        country_name = currency_countries[selected_currency_code]
        st.write(f'Predicted Exchange Rates for {country_name} ({selected_currency_code}) in Rs:')
        prediction_df = pd.DataFrame({'Date': prediction_dates, f'Predicted Exchange Rate ({country_name})': predictions})
        st.write(prediction_df)

        # Show historical exchange rate chart if selected
        if show_historical_chart:
            currency_data = data.loc[:, [selected_currency_code]]
            plot_historical_rates(currency_data, selected_currency_code)
