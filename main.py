import base64
import io
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from app_utils import ValidationError, normalize_analysis_request
from valuation_service import resolve_analysis
from zillow_data import ZillowDataError


app = Flask(__name__, static_url_path='/static')


def image_html_from_figure(fig, alt_text):
    plot_bytes_io = io.BytesIO()
    fig.savefig(plot_bytes_io, format='png', bbox_inches='tight')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{plot_base64}" alt="{alt_text}" />'


def market_history_plot(analysis):
    market = analysis['market']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(market['sfr_series'].index, market['sfr_series'].values, color='black', label='City SFR ZHVI')

    if market['bedroom_series'] is not None:
        ax.plot(
            market['bedroom_series'].index,
            market['bedroom_series'].values,
            color='firebrick',
            label=market['primary_label'],
        )

    valuation_price = analysis['valuation'].get('price')
    if valuation_price:
        latest_date = market['primary_series'].index[-1]
        ax.scatter([latest_date], [valuation_price], color='royalblue', zorder=3, label='Estimated subject value')
        ax.axhline(valuation_price, ls='dashed', alpha=0.25, color='royalblue')

    ax.set_title(f"Market history for {market['location']}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated home value (USD)')
    ax.legend()
    return image_html_from_figure(fig, 'Market history chart')


def forecast_plot(analysis, period_months):
    try:
        from prophet import Prophet
    except Exception as exc:
        analysis['warnings'].append(f'Prophet forecast skipped because Prophet could not be imported: {exc}')
        return None

    series = analysis['market']['primary_series'].dropna()
    if len(series) < 24:
        analysis['warnings'].append('Prophet forecast skipped because fewer than 24 months of market data were available.')
        return None

    model_df = series.reset_index()
    model_df.columns = ['ds', 'y']
    try:
        model = Prophet(yearly_seasonality=True)
        model_fit = model.fit(model_df)
        future = model_fit.make_future_dataframe(periods=period_months, freq='ME')
        pred = model_fit.predict(future)
    except Exception as exc:
        analysis['warnings'].append(f'Prophet forecast skipped because model fitting failed: {exc}')
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(pred, ax=ax)
    ax.set_title(f"{analysis['market']['primary_label']} market-index forecast")
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated home value (USD)')
    return image_html_from_figure(fig, 'Market forecast chart')


@app.template_filter('currency')
def currency(value):
    if value is None:
        return 'Not available'
    return f'${float(value):,.0f}'


@app.template_filter('number')
def number(value):
    if value is None:
        return 'Not available'
    return f'{float(value):,.0f}'


@app.template_filter('percent')
def percent(value):
    if value is None:
        return 'Not available'
    return f'{float(value):,.1f}%'


@app.template_filter('decimal')
def decimal(value):
    if value is None:
        return 'Not available'
    return f'{float(value):,.2f}'


@app.route('/')
def index():
    return render_template('housing.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        analysis_request = normalize_analysis_request(request.form)
        analysis = resolve_analysis(analysis_request)
        market_plot_html = market_history_plot(analysis)
        forecast_plot_html = forecast_plot(analysis, analysis_request['period_months'])
    except (ValidationError, ZillowDataError, RuntimeError) as exc:
        return render_template('result.html', error=f'ERROR: {exc}')

    return render_template(
        'result.html',
        analysis=analysis,
        market_plot_html=market_plot_html,
        forecast_plot_html=forecast_plot_html,
        period=analysis_request['period'],
        period_unit=analysis_request['period_unit'],
    )


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG') == '1')
