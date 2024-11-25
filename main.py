import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# Következő teendők.
# A model kész viszont a nem egyenlő számú buy és sell miatt az egyik érték dominál a tanulás során. Ezt finomhangolni kell esetlegesen balance függvénnyel vagy más tanulási rátával
#
#


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

#Create the Signal logic
def apply_total_signal(data, rsi_threshold_low=30, rsi_threshold_high=70, bb_width_threshold=0.0015):
    # Kezdeti feltételek
    data['totalSignal'] = 0

    # Feltételek kiszámítása
    prev_candle_closes_below_bb = data['close'].shift(1) < data['bbl'].shift(1)
    prev_rsi_below_thr = data['rsi'].shift(1) < rsi_threshold_low
    closes_above_prev_high = data['close'] > data['high'].shift(1)
    bb_width_greater_threshold = data['bb_width'] > bb_width_threshold

    # Buy jelzés
    buy_signal = (
        prev_candle_closes_below_bb &
        prev_rsi_below_thr &
        closes_above_prev_high &
        bb_width_greater_threshold
    )

    # Sell jelzés
    prev_candle_closes_above_bb = data['close'].shift(1) > data['bbh'].shift(1)
    prev_rsi_above_thr = data['rsi'].shift(1) > rsi_threshold_high
    closes_below_prev_low = data['close'] < data['low'].shift(1)

    sell_signal = (
        prev_candle_closes_above_bb &
        prev_rsi_above_thr &
        closes_below_prev_low &
        bb_width_greater_threshold
    )

    # Összesített jelzés
    data.loc[buy_signal, 'totalSignal'] = 2
    data.loc[sell_signal, 'totalSignal'] = 1

    return data


#define the entry points
def pointpos(x):
    if x['totalSignal']==2:
        return x['low']-1e-4
    elif x['totalSignal']==1:
        return x['high']+1e-4
    else:
        return np.nan

apply_total_signal(data=data, rsi_threshold_low=30, rsi_threshold_high=70, bb_width_threshold=0.001)
data['pointpos'] = data.apply(lambda row: pointpos(row), axis=1)

########################### Create the model ####################################

# Csak a Buy és Sell jelek megtartása
filtered_data = data[data['totalSignal'].isin([1, 2])]

X = filtered_data[['open', 'high', 'low', 'close', 'rsi', 'bbl', 'bbm', 'bbh', 'bb_width']]
y = filtered_data['totalSignal'] - 1  # Átalakítás 0 (Sell) és 1 (Buy) címkékre

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neurális hálózat módosítása
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid aktiváció egy bináris kimenetre

# Modell kompilálása
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Early stopping beállítása
early_stopping = EarlyStopping(
    monitor='val_loss',  # Mit figyeljünk: itt a validációs veszteség
    patience=5,          # Hány epizód után álljon meg, ha nincs javulás
    restore_best_weights=True  # A legjobb modell súlyainak visszaállítása
)

# Modell tanítása
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Előrejelzés készítése
y_pred_nn = model.predict(X_test)
y_pred_classes = (y_pred_nn > 0.5).astype(int)  # Bináris osztályozás küszöbértékkel

# Eredmény kiértékelése
print(classification_report(y_test, y_pred_classes))

cm = confusion_matrix(y_test, y_pred_classes)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Buy'], yticklabels=['Sell', 'Buy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

####################################### PLOTING #################################
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
                         line=dict(color='blue', width=1),
                         name="RSI"),
              row=2, col=1)

# A jelölés hozzáadása
fig.add_trace(go.Scatter(x=data_last_10_days.index, y=data_last_10_days['pointpos'], mode="markers",
                         marker=dict(size=8, color="MediumPurple"),
                         name="entry"),
              row=1, col=1)

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
    height=1400,
    sliders=[]
)


# Diagram megjelenítése
fig.show()
