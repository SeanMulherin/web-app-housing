import base64
import io
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from app_utils import ValidationError, normalize_analysis_request
from valuation_service import resolve_analysis
from zillow_data import ZillowDataError


app = Flask(__name__, static_url_path='/static')

ALLOWED_API_ORIGINS = {
    'https://housing-market-lab.sean-mulherin.chatgpt.site',
    'http://localhost:4174',
    'http://127.0.0.1:4174',
}


@app.after_request
def add_api_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_API_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Vary'] = 'Origin'
    return response


def image_html_from_figure(fig, alt_text):
    plot_bytes_io = io.BytesIO()
    fig.savefig(plot_bytes_io, format='png', bbox_inches='tight')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{plot_base64}" alt="{alt_text}" />'


def _short_money(value):
    return rf'\${float(value) / 1000:,.0f}k'


def _price_per_sqft(price, square_footage):
    if not price or not square_footage:
        return None
    square_footage = float(square_footage)
    if square_footage <= 0:
        return None
    return float(price) / square_footage


def _short_property_label(comparable):
    address = comparable.get('formatted_address') or comparable.get('address_line_1') or 'Comparable property'
    address_parts = [part.strip() for part in address.split(',') if part.strip()]
    label = address_parts[0] if address_parts else address.strip()
    if len(address_parts) > 1 and address_parts[1].lower().startswith('lot '):
        label = f'{label}, {address_parts[1]}'
    return label if len(label) <= 34 else f'{label[:31]}...'


def comparable_values_plot(analysis):
    subject_value = analysis['valuation'].get('price')
    comparables = [
        comparable for comparable in analysis.get('comparables', [])
        if comparable.get('price') is not None
    ]
    if not subject_value or not comparables:
        return None

    rows = []
    for comparable in comparables:
        price = float(comparable['price'])
        rows.append({
            'label': _short_property_label(comparable),
            'status': comparable.get('status') or 'Unknown',
            'price': price,
            'price_per_sqft': _price_per_sqft(price, comparable.get('square_footage')),
            'distance': comparable.get('distance'),
            'fit': comparable.get('correlation'),
        })

    rows = sorted(rows, key=lambda row: row['price'])
    prices = np.array([row['price'] for row in rows])
    labels = [row['label'] for row in rows]
    colors = ['#257f80' if row['status'] == 'Active' else '#a5abb3' for row in rows]
    y_values = np.arange(len(rows))

    subject_low = analysis['valuation'].get('price_range_low')
    subject_high = analysis['valuation'].get('price_range_high')
    x_values_for_bounds = list(prices) + [float(subject_value)]
    if subject_low:
        x_values_for_bounds.append(float(subject_low))
    if subject_high:
        x_values_for_bounds.append(float(subject_high))

    min_value = min(x_values_for_bounds)
    max_value = max(x_values_for_bounds)
    value_span = max(max_value - min_value, max_value * 0.2, 1)
    x_min = max(0, min_value - value_span * 0.12)
    x_max = max_value + max(value_span * 0.55, max_value * 0.14, 120000)

    fig_height = min(11, max(5.5, 2.8 + 0.42 * len(rows)))
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if subject_low and subject_high:
        ax.axvspan(float(subject_low), float(subject_high), color='#f2c94c', alpha=0.24, zorder=0)
    ax.axvline(float(subject_value), color='#b8322a', linewidth=3, zorder=3)
    ax.barh(y_values, prices, color=colors, height=0.64, zorder=2)

    ax.set_yticks(y_values)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Comparable price', fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(lambda value, position: _short_money(value))
    ax.grid(axis='x', color='#ececec', linewidth=1, zorder=1)
    ax.tick_params(axis='y', length=0)

    label_offset = max(value_span * 0.018, 9000)
    for index, row in enumerate(rows):
        diff = (row['price'] - float(subject_value)) / float(subject_value) * 100
        diff_label = f'+{diff:.0f}%' if diff >= 0 else f'{diff:.0f}%'
        label_parts = [_short_money(row['price']), diff_label]
        if row['price_per_sqft'] is not None:
            label_parts.append(rf'\${row["price_per_sqft"]:,.0f}/sf')
        if row['distance'] is not None:
            label_parts.append(f'{float(row["distance"]):.2f} mi')
        if row['fit'] is not None:
            label_parts.append(f'fit {float(row["fit"]):.2f}')
        ax.text(
            row['price'] + label_offset,
            index,
            '  '.join(label_parts),
            va='center',
            fontsize=9.2,
            color='#505050',
            clip_on=False,
        )

    median_price = float(np.median(prices))
    subject_ppsf = _price_per_sqft(subject_value, analysis.get('subject', {}).get('square_footage'))
    comparable_ppsf_values = [row['price_per_sqft'] for row in rows if row['price_per_sqft'] is not None]
    median_ppsf = float(np.median(comparable_ppsf_values)) if comparable_ppsf_values else None

    subject_summary = f'Subject estimate: {_short_money(subject_value)}'
    if subject_ppsf is not None:
        subject_summary = f'{subject_summary} (\\${subject_ppsf:,.0f}/sf)'
    if subject_low and subject_high:
        subject_summary = f'{subject_summary}, range {_short_money(subject_low)}-{_short_money(subject_high)}'
    comparable_summary = f'Comp median: {_short_money(median_price)}'
    if median_ppsf is not None:
        comparable_summary = f'{comparable_summary} (\\${median_ppsf:,.0f}/sf)'

    ax.set_title(
        'Comparable Properties vs Subject Estimate',
        loc='left',
        fontsize=20,
        fontweight='bold',
        color='#4d4d4d',
        pad=42,
    )
    ax.text(
        0,
        1.025,
        f'{subject_summary} | {comparable_summary}',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=11.5,
        color='#666666',
        clip_on=False,
    )

    legend_items = [
        Patch(facecolor='#257f80', label='Active comp'),
        Patch(facecolor='#a5abb3', label='Inactive comp'),
        Patch(facecolor='#f2c94c', alpha=0.24, label='Subject AVM range'),
        Line2D([0], [0], color='#b8322a', lw=3, label='Subject estimate'),
    ]
    ax.legend(handles=legend_items, loc='lower right', frameon=False, ncol=2, fontsize=9.5)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.24, right=0.94, top=0.82, bottom=0.12)
    return image_html_from_figure(fig, 'Comparable properties compared with subject estimate')


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


def _serialize_series(series):
    if series is None:
        return []
    return [
        {'date': timestamp.date().isoformat(), 'value': float(value)}
        for timestamp, value in series.dropna().items()
    ]


def _serialize_analysis(analysis):
    market = analysis['market']
    return {
        'subject': analysis['subject'],
        'valuation': analysis['valuation'],
        'comparables': analysis.get('comparables', []),
        'warnings': analysis.get('warnings', []),
        'market': {
            'location': market['location'],
            'primary_label': market['primary_label'],
            'latest_value': market['latest_value'],
            'latest_date': market['latest_date'],
            'lookback_percent_change': market['lookback_percent_change'],
            'sfr_latest_value': market.get('sfr_latest_value', market['latest_value']),
            'bedroom_latest_value': market.get('bedroom_latest_value'),
            'sfr_series': _serialize_series(market['sfr_series']),
            'bedroom_series': _serialize_series(market['bedroom_series']),
        },
    }


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
        comparable_plot_html = comparable_values_plot(analysis)
        forecast_plot_html = forecast_plot(analysis, analysis_request['period_months'])
    except (ValidationError, ZillowDataError, RuntimeError) as exc:
        return render_template('result.html', error=f'ERROR: {exc}')

    return render_template(
        'result.html',
        analysis=analysis,
        market_plot_html=market_plot_html,
        comparable_plot_html=comparable_plot_html,
        forecast_plot_html=forecast_plot_html,
        period=analysis_request['period'],
        period_unit=analysis_request['period_unit'],
    )


@app.route('/api/analysis', methods=['POST', 'OPTIONS'])
def api_analysis():
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        payload = request.get_json(silent=True) or request.form
        analysis_request = normalize_analysis_request(payload)
        analysis = resolve_analysis(analysis_request)
    except (ValidationError, ZillowDataError, RuntimeError) as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(_serialize_analysis(analysis))


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG') == '1')
