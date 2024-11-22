import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CSV fájl beolvasása
file_path = "EURUSD_1H_2020-2024.csv"  # Állítsd be a fájl elérési útját
data = pd.read_csv(file_path)

# Időbélyegek kezelése
data['time'] = pd.to_datetime(data['time'])  # Átalakítás datetime formátumba
data.set_index('time', inplace=True)  # Beállítjuk az időt indexként

# RSI és Bollinger Band számítása
data.ta.rsi(append=True, length=14)
data.ta.bbands(append=True, length=30, std=2)

# Átnevezés a könnyebb használhatóság érdekében
data.rename(columns={'real_volume': 'volume', 'BBL_30_2.0': 'bbl', 'BBM_30_2.0': 'bbm', 'BBU_30_2.0': 'bbh', 'RSI_14': 'rsi'}, inplace=True)
data['bb_width'] = (data['bbh'] - data['bbl']) / data['bbm']
# Csak az utolsó 240 adat kiválasztása (utolsó 10 nap, ha órás adataink vannak)
data_last_10_days = data.tail(2400)


#Create the Signal logic
def apply_total_signal(data, rsi_threshold_low=30, rsi_threshold_high=70, bb_width_threshold = 0.0015):
    # Initialize the 'TotalSignal' column
    data['TotalSignal'] = 0

    for i in range(1, len(data)):
        # Previous candle conditions
        prev_candle_closes_below_bb = data['close'].iloc[i-1] < data['bbl'].iloc[i-1]
        prev_rsi_below_thr = data['rsi'].iloc[i-1] < rsi_threshold_low
        # Current candle conditions
        closes_above_prev_high = data['close'].iloc[i] > data['high'].iloc[i-1]
        bb_width_greater_threshold = data['bb_width'].iloc[i] > bb_width_threshold

        # Combine conditions
        if (prev_candle_closes_below_bb and
            prev_rsi_below_thr and
            closes_above_prev_high and
            bb_width_greater_threshold):
            data.at[i, 'TotalSignal'] = 2  # Set the buy signal for the current candle

        # Previous candle conditions
        prev_candle_closes_above_bb = data['Close'].iloc[i-1] > data['bbh'].iloc[i-1]
        prev_rsi_above_thr = data['rsi'].iloc[i-1] > rsi_threshold_high
        # Current candle conditions
        closes_below_prev_low = data['close'].iloc[i] < data['low'].iloc[i-1]
        bb_width_greater_threshold = data['bb_width'].iloc[i] > bb_width_threshold

        # Combine conditions
        if (prev_candle_closes_above_bb and
            prev_rsi_above_thr and
            closes_below_prev_low and
            bb_width_greater_threshold):
            data.at[i, 'TotalSignal'] = 1  # Set the sell signal for the current candle


    return data





# Create a plot with 2 rows
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], 
                    subplot_titles=('EUR/USD Gyertyadiagram', 'RSI'),
                    vertical_spacing=0.25)

# Gyertyadiagram hozzáadása az első sorhoz
fig.add_trace(go.Candlestick(x=data_last_10_days.index,
                             open=data_last_10_days['open'],
                             high=data_last_10_days['high'],
                             low=data_last_10_days['low'],
                             close=data_last_10_days['close'],
                             name='Candlesticks'),
              row=1, col=1)

# Bollinger Bands háttér kitöltése
fig.add_trace(go.Scatter(x=data_last_10_days.index, y=data_last_10_days['bbh'],
                         line=dict(color='red', width=1),
                         name="BBU"),
              row=1, col=1)

fig.add_trace(go.Scatter(x=data_last_10_days.index, y=data_last_10_days['bbl'],
                         line=dict(color='green', width=1),
                         name="BBL"),
              row=1, col=1)

# Kitöltés a Bollinger Bands közötti területen (BBL és BBM között) zöld
fig.add_trace(go.Scatter(x=data_last_10_days.index, 
                         y=data_last_10_days['bbm'],
                         fill='tonexty',
                         fillcolor='rgba(255, 0, 0, 0.1)',  # Piros háttér a BBL és BBM között
                         line=dict(width=0), name="BBL-BBM Fill", showlegend=False),
              row=1, col=1)

# Kitöltés a Bollinger Bands közötti területen (BBH és BBM között) piros
fig.add_trace(go.Scatter(x=data_last_10_days.index, 
                         y=data_last_10_days['bbh'],
                         fill='tonexty',
                         fillcolor='rgba(0, 255, 0, 0.1)',  #Zöld háttér a BBH és BBM között
                         line=dict(width=0), name="BBH-BBM Fill", showlegend=False),
              row=1, col=1)

# Bollinger Bands középvonal hozzáadása
fig.add_trace(go.Scatter(x=data_last_10_days.index, y=data_last_10_days['bbm'],
                         line=dict(color='blue', width=1),
                         name="BBM"),
              row=1, col=1)

# RSI hozzáadása a második sorhoz
fig.add_trace(go.Scatter(x=data_last_10_days.index, y=data_last_10_days['rsi'],
                         line=dict(color='blue', width=1),
                         name="RSI"),
              row=2, col=1)
# Vízszintes vonalak hozzáadása az RSI grafikonhoz (30 és 70 szint)
fig.add_shape(
    type="line",
    x0=data_last_10_days.index.min(),
    x1=data_last_10_days.index.max(),
    y0=30,
    y1=30,
    line=dict(color="green", width=2, dash="solid"),
    xref="x2",  # Az RSI subplot x tengelyére vonatkozik
    yref="y2",  # Az RSI subplot y tengelyére vonatkozik
)
fig.add_shape(
    type="line",
    x0=data_last_10_days.index.min(),
    x1=data_last_10_days.index.max(),
    y0=70,
    y1=70,
    line=dict(color="red", width=2, dash="solid"),
    xref="x2",  # Az RSI subplot x tengelyére vonatkozik
    yref="y2",  # Az RSI subplot y tengelyére vonatkozik
)

# Layout beállítások
fig.update_layout(
    title='EUR/USD Gyertyadiagram és RSI',
    xaxis=dict(title='Date'),
    xaxis2=dict(title='Date'),
    yaxis=dict(title='Price'),
    yaxis2=dict(title='RSI', range=[20, 90], tickvals=[20, 30, 40, 50, 60, 70, 80, 90]),
    showlegend=True,
    width=1980,
    height=1024,
    sliders=[]
)

# Diagram megjelenítése
fig.show()
