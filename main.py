import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from prophet import Prophet
import base64
import io

app = Flask(__name__, static_url_path='/static')

url='https://drive.google.com/file/d/1xx9u956zag8992-Vf5hFgKg9JkqvVh1J/view?usp=drive_link'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
h = pd.read_csv(url)
h = h.drop(['RegionID', 'SizeRank', 'RegionType', 'StateName', 'Metro', 'CountyName'], axis=1)
h.insert(0, 'Location', h['RegionName'] + ', ' + h['State'])
h = h.drop(['RegionName', 'State'], axis=1)
h = np.transpose(h)
h.columns = h.iloc[0, ]
h = h.drop('Location', axis=0)
h.index = pd.to_datetime(h.index)
h.index.name = 'date'
h = h.astype(float)
country_avg = h.mean(axis=1)


def house_fcst(name, period):
    if period_unit == "year":
        period = period*12

    df = h.loc[:, name].to_frame()
    df.columns = ['y']
    obs_price = df.y[-1]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df.y, color='firebrick', label='City AVG')
    ax1.plot(country_avg, color='black', label='Country AVG')
    ax1.set_title(f"AVG Monthly House Price - {name}", size='x-large', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('AVG Selling Price (USD)')
    ax1.plot(df.index[-1], df.y[-1], 'bo')
    ax1.axhline(df.y[-1], ls='dashed', alpha=0.2, color='black')
    ax1.axvline(df.index[-1], ls='dashed', alpha=0.2, color='black')
    ax1.text(df.index[-1] + timedelta(500), df.y[-1], f'${obs_price:.2f}')
    ax1.legend()


    plot_bytes_io1 = io.BytesIO()
    plt.savefig(plot_bytes_io1, format='png')
    plot_bytes_io1.seek(0)
    plot_base641 = base64.b64encode(plot_bytes_io1.getvalue()).decode()
    plot_html1 = f'<img src="data:image/png;base64,{plot_base641}" alt="plot" />'

    plt.close()


    df = df.reset_index()
    df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True)
    model_fit = model.fit(df)
    future = model_fit.make_future_dataframe(periods=period, freq='M')
    pred = model_fit.predict(future)

    fcst_price = pred.yhat[len(pred) - 1]
    future_pctch = 100 * (pred.trend.iloc[-1] - pred.trend[len(pred) - period - 1]) / pred.trend[
        len(pred) - period - 1]
    past_decade_pctch = 100 * (df.y.iloc[-1] - df.y.iloc[-12 * 10]) / df.y.iloc[-12 * 10]

    future_pctch_bool = "decrease"
    past_decade_pctch_bool = "decreased"
    if fcst_price > 0:
        future_pctch_bool = 'increase'
    if future_pctch > 0:
        past_decade_pctch_bool = 'increased'

    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(pred, ax=ax)
    ax.plot(country_avg, color='red', label='Observed Country AVG')
    ax.set_title(f'Forecast Model of House Prices in {name}')
    ax.plot(pred.ds.iloc[-1], fcst_price, 'bo')
    ax.axhline(fcst_price, ls='dashed', alpha=0.2, color='black')
    ax.axvline(pred.ds.iloc[-1], ls='dashed', alpha=0.2, color='black')
    ax.text(pred.ds.iloc[-1] + timedelta(days=500), fcst_price, f"${fcst_price:.2f}")
    ax.set_xlabel('Date')
    ax.set_ylabel('AVG Selling Price (USD)')
    ax.legend()

    plot_bytes_io = io.BytesIO()
    plt.savefig(plot_bytes_io, format='png')
    plot_bytes_io.seek(0)
    plot_base64 = base64.b64encode(plot_bytes_io.getvalue()).decode()
    plot_html = f'<img src="data:image/png;base64,{plot_base64}" alt="plot" />'

    plt.close()

    components_fig = model.plot_components(pred, figsize=(10, 7))
    components_bytes_io = io.BytesIO()
    components_fig.savefig(components_bytes_io, format='png')
    components_bytes_io.seek(0)
    components_base64 = base64.b64encode(components_bytes_io.getvalue()).decode()
    components_html = f'<img src="data:image/png;base64,{components_base64}" alt="components plot" />'

    plt.close(components_fig)

    return (f"Housing prices in {name} have {past_decade_pctch_bool} by {past_decade_pctch:.2f}% in the past decade "
            f"and are predicted to {future_pctch_bool} by {future_pctch:.2f}% in the next {period} {period_unit}(s).", 
            plot_html1, plot_html, components_html)


@app.route('/')
def index():
    return render_template('housing.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    city = request.form['city']
    city = city.strip()
    state = request.form['state']
    location = city.title() + ", " + state.upper()
    period = float(request.form['period'])
    period_int = int(period)
    period_unit = request.form['period_unit']

    if location not in h.columns:
        result = f"ERROR: {location} IS EITHER MISSPELLED OR NOT IN THE DATABASE. PLEASE GO BACK A PAGE AND TRY AGAIN."
        return render_template('result.html', result=result)
    elif period_int <= 0:
        result = f"ERROR: FORECAST LENGTH MUST BE A POSITIVE INTEGER > 0. YOU INPUT {period} AS YOUR FORECAST LENGTH. PLEASE GO BACK A PAGE AND TRY AGAIN."
        return render_template('result.html', result=result)
    else:
        result,plot_html1, plot_html, components_html = house_fcst(location, period_int)
        return render_template('result.html', result=result, plot_html1=plot_html1, plot_html=plot_html, components_html=components_html)


if __name__ == "__main__":
    app.run(debug=True)
