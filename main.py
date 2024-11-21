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

# Csak az utolsó 240 adat kiválasztása (utolsó 10 nap, ha órás adataink vannak)
data_last_10_days = data.tail(240)

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
                         line=dict(color='blue', width=2),
                         name="RSI"),
              row=2, col=1)

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
