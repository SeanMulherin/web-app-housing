import base64
import io
import os
from datetime import timedelta
from functools import lru_cache

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from prophet import Prophet

from app_utils import (
    ValidationError,
    direction_word,
    normalize_location,
    percent_change,
    period_to_months,
    validate_forecast_period,
)


FILE_ID = '1tEZqEQc-SNtde97-T9pyt5fHG0OiaxPY'
DATA_URL = f'https://drive.google.com/uc?id={FILE_ID}'

app = Flask(__name__, static_url_path='/static')


@lru_cache(maxsize=1)
def get_housing_data():
    try:
        housing = pd.read_csv(DATA_URL)
    except Exception as exc:
        raise RuntimeError('Housing data could not be loaded right now. Please try again later.') from exc

    required_columns = {'RegionID', 'SizeRank', 'RegionType', 'StateName', 'Metro', 'CountyName', 'RegionName', 'State'}
    missing = required_columns.difference(housing.columns)
    if missing:
        raise RuntimeError('Housing data did not contain the expected Zillow columns.')

    housing = housing.drop(['RegionID', 'SizeRank', 'RegionType', 'StateName', 'Metro', 'CountyName'], axis=1)
    housing.insert(0, 'Location', housing['RegionName'] + ', ' + housing['State'])
    housing = housing.drop(['RegionName', 'State'], axis=1)
    housing = np.transpose(housing)
    housing.columns = housing.iloc[0, ]
    housing = housing.drop('Location', axis=0)
    housing.index = pd.to_datetime(housing.index)
    housing.index.name = 'date'
    housing = housing.astype(float)

    return housing, housing.mean(axis=1)


def image_html_from_current_plot(alt_text):
    plot_bytes_io = io.BytesIO()
    plt.savefig(plot_bytes_io, format='png', bbox_inches='tight')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plt.close()
    return f'<img src="data:image/png;base64,{plot_base64}" alt="{alt_text}" />'


def image_html_from_figure(fig, alt_text):
    plot_bytes_io = io.BytesIO()
    fig.savefig(plot_bytes_io, format='png', bbox_inches='tight')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{plot_base64}" alt="{alt_text}" />'


def house_fcst(name, period, period_unit):
    period_months = period_to_months(period, period_unit)
    housing, country_avg = get_housing_data()

    if name not in housing.columns:
        raise ValidationError(f'{name} is either misspelled or not in the database.')

    df = housing.loc[:, name].dropna().to_frame()
    df.columns = ['y']
    if len(df) < 24:
        raise RuntimeError(f'Not enough housing data was returned for {name}.')

    obs_price = float(df.y.iloc[-1])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df.y, color='firebrick', label='City AVG')
    ax1.plot(country_avg, color='black', label='Country AVG')
    ax1.set_title(f'AVG Monthly House Price - {name}', size='x-large', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('AVG Selling Price (USD)')
    ax1.plot(df.index[-1], obs_price, 'bo')
    ax1.axhline(obs_price, ls='dashed', alpha=0.2, color='black')
    ax1.axvline(df.index[-1], ls='dashed', alpha=0.2, color='black')
    ax1.text(df.index[-1] + timedelta(500), obs_price, f'${obs_price:.2f}')
    ax1.legend()
    plot_html1 = image_html_from_current_plot('Observed home prices')

    model_df = df.reset_index()
    model_df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True)
    model_fit = model.fit(model_df)
    future = model_fit.make_future_dataframe(periods=period_months, freq='ME')
    pred = model_fit.predict(future)

    fcst_price = float(pred.yhat.iloc[-1])
    future_reference = float(pred.trend.iloc[-period_months - 1])
    future_pctch = percent_change(float(pred.trend.iloc[-1]), future_reference)

    past_reference_index = max(0, len(model_df) - 121)
    past_decade_pctch = percent_change(float(model_df.y.iloc[-1]), float(model_df.y.iloc[past_reference_index]))

    future_pctch_bool = direction_word(future_pctch, tense='present')
    past_decade_pctch_bool = direction_word(past_decade_pctch, tense='past')

    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(pred, ax=ax)
    ax.plot(country_avg, color='red', label='Observed Country AVG')
    ax.set_title(f'Forecast Model of House Prices in {name}')
    ax.plot(pred.ds.iloc[-1], fcst_price, 'bo')
    ax.axhline(fcst_price, ls='dashed', alpha=0.2, color='black')
    ax.axvline(pred.ds.iloc[-1], ls='dashed', alpha=0.2, color='black')
    ax.text(pred.ds.iloc[-1] + timedelta(days=500), fcst_price, f'${fcst_price:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('AVG Selling Price (USD)')
    ax.legend()
    plot_html = image_html_from_current_plot('Forecasted home prices')

    components_fig = model.plot_components(pred, figsize=(10, 7))
    components_html = image_html_from_figure(components_fig, 'Forecast components plot')

    return (
        f'Housing prices in {name} have {past_decade_pctch_bool} by {past_decade_pctch:.2f}% in the past decade '
        f'and are predicted to {future_pctch_bool} by {future_pctch:.2f}% in the next {period} {period_unit}(s).',
        plot_html1,
        plot_html,
        components_html,
    )


@app.route('/')
def index():
    return render_template('housing.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        location = normalize_location(request.form.get('city', ''), request.form.get('state', ''))
        period_unit = request.form.get('period_unit', '')
        period = validate_forecast_period(request.form.get('period', ''), period_unit)
        housing, _ = get_housing_data()
    except (ValidationError, RuntimeError) as exc:
        return render_template('result.html', result=f'ERROR: {exc}')

    if location not in housing.columns:
        result = f'ERROR: {location} IS EITHER MISSPELLED OR NOT IN THE DATABASE. PLEASE GO BACK A PAGE AND TRY AGAIN.'
        return render_template('result.html', result=result)

    try:
        result, plot_html1, plot_html, components_html = house_fcst(location, period, period_unit)
    except Exception as exc:
        app.logger.exception('Housing forecast failed')
        return render_template('result.html', result=f'ERROR: {exc}')

    return render_template('result.html', result=result, plot_html1=plot_html1, plot_html=plot_html, components_html=components_html)


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG') == '1')
