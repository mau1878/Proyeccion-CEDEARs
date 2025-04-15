import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np

# **Important:** `st.set_page_config` must be the first Streamlit command
st.set_page_config(
  page_title="Proyección de CEDEARs Ajustados por Inflación",
  layout="wide"
)

# Importación de la función de ajuste de splits
from inflacion import ajustar_precios_por_splits  # Asegúrate de que este módulo está limpio

# Título de la aplicación
st.title("Proyección del rendimiento de CEDEARs ajustados por inflación")

# ### Carga de datos de IPC con caché
@st.cache_data
def load_cpi_data():
  try:
      cpi = pd.read_csv('https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nargentina2.csv')
      cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
      cpi.set_index('Date', inplace=True)
      # Asegurar que 'CPI_MoM' esté en formato decimal (ejemplo: 5% como 0.05)
      if cpi['CPI_MoM'].max() > 10:
          cpi['CPI_MoM'] = cpi['CPI_MoM'] / 100
      cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
      daily_cpi = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
      return daily_cpi
  except FileNotFoundError:
      st.error("El archivo 'inflaciónargentina2.csv' no se encontró.")
      st.stop()
  except Exception as e:
      st.error(f"Error al cargar los datos de inflación: {e}")
      st.stop()

# ### Carga de ratios de CEDEAR con caché
@st.cache_data
def load_cedear_ratios():
  try:
      ratios = pd.read_csv('ratioscedears.csv')
      return ratios.set_index('CEDEAR')
  except FileNotFoundError:
      st.error("El archivo 'ratioscedears.csv' no se encontró.")
      st.stop()
  except Exception as e:
      st.error(f"Error al cargar las ratios de CEDEARs: {e}")
      st.stop()

# Carga de datos
daily_cpi = load_cpi_data()
cedear_ratios = load_cedear_ratios()

# ### Barra lateral para entradas de usuario
st.sidebar.header("Parámetros de Proyección")

# Entradas para fechas de inicio y fin
start_date_default = datetime(2022, 1, 1)
end_date_default = datetime.now() + timedelta(days=365)

start_date = st.sidebar.date_input(
  "Fecha de compra de CEDEARs (Inicio)",
  value=start_date_default,
  min_value=datetime(2000, 1, 1).date(),
  max_value=datetime.now().date()
)

end_date = st.sidebar.date_input(
  "Fecha de finalización de la predicción",
  value=end_date_default.date(),
  min_value=(start_date + timedelta(days=1))
)

if end_date <= start_date:
  st.sidebar.error("La fecha de finalización debe ser posterior a la fecha de inicio.")
  st.stop()

# Convertir end_date a pd.Timestamp
end_date_ts = pd.Timestamp(end_date)

# **Overhaul: Añadiendo Entradas para Start y End Inflation Rates y Exchange Rates**

st.sidebar.markdown("### Progresión de Inflación y Tipo de Cambio")

# Entradas para tasa de inflación mensual inicial y final
start_inflation_rate = st.sidebar.number_input(
  "Tasa de inflación mensual inicial (%)",
  min_value=0.0,
  max_value=100.0,
  value=3.5,
  step=0.1
) / 100  # Convertir a decimal

end_inflation_rate = st.sidebar.number_input(
  "Tasa de inflación mensual final (%)",
  min_value=0.0,
  max_value=100.0,
  value=0.5,
  step=0.1
) / 100  # Convertir a decimal

# Nuevo: Factor de aceleración para Inflación
inflation_acceleration = st.sidebar.slider(
  "Factor de aceleración para Inflación",
  min_value=0.1,
  max_value=3.0,
  value=2.0,
  step=0.1,
  help="Controla la aceleración de la progresión de inflación. >1: acelera hacia el final, <1: decelera."
)

# Entradas para tipo de cambio inicial y final
start_exchange_rate = st.sidebar.number_input(
  "Tipo de cambio inicial (USD/ARS)",
  min_value=0.0,
  value=700.0,
  step=10.0
)

end_exchange_rate = st.sidebar.number_input(
  "Tipo de cambio final (USD/ARS)",
  min_value=0.0,
  value=900.0,
  step=10.0
)

# Nuevo: Factor de aceleración para Tipo de Cambio
exchange_acceleration = st.sidebar.slider(
  "Factor de aceleración para Tipo de Cambio",
  min_value=0.1,
  max_value=3.0,
  value=2.0,
  step=0.1,
  help="Controla la aceleración de la progresión del tipo de cambio. >1: acelera hacia el final, <1: decelera."
)

# Entrada para tasa de crecimiento del activo subyacente
growth_rate_underlying_asset = st.sidebar.number_input(
  "Tasa de crecimiento anual del activo subyacente (%)",
  min_value=-100.0,
  value=10.0,
  step=0.1
) / 100

# Entradas para tickers de CEDEAR y activo subyacente
ticker = st.sidebar.text_input(
  "Ingresar CEDEAR (por ejemplo, SPY.BA):",
  value="SPY.BA"
).upper()

underlying_ticker = st.sidebar.text_input(
  "Ingresar ticker del activo subyacente (por ejemplo, SPY):",
  value="SPY"
).upper()

# ### Simulaciones de Tipo de Cambio
st.sidebar.subheader("Simulaciones de Tipo de Cambio (USD/ARS)")

# Eventos de tipo de cambio por defecto con fechas como pd.Timestamp
default_exchange_events = [
  {"Fecha": pd.Timestamp("2024-06-01"), "USD/ARS": 750.0},
  {"Fecha": pd.Timestamp("2024-12-01"), "USD/ARS": 850.0},
]

# Editor de datos para eventos de tipo de cambio
exchange_events = st.sidebar.data_editor(
  default_exchange_events,
  column_config={
      "Fecha": st.column_config.DateColumn("Fecha de Evento"),
      "USD/ARS": st.column_config.NumberColumn("Tipo de Cambio (USD/ARS)", format="%.2f"),
  },
  num_rows="dynamic",
  use_container_width=True
)

# Convertir todas las fechas a pd.Timestamp para consistencia
for event in exchange_events:
  event["Fecha"] = pd.Timestamp(event["Fecha"])

# ### Función para crear una progresión no lineal (ej. cuadrática)
def create_progressive_series(start_value, end_value, num_steps, acceleration=1.0):
  """
  Crea una serie de valores que progresa de start_value a end_value de manera no lineal.
  
  Parámetros:
  - start_value (float): Valor inicial.
  - end_value (float): Valor final.
  - num_steps (int): Número de pasos (días).
  - acceleration (float): Factor de aceleración. >1 acelera hacia el final, <1 decelera.
  
  Retorna:
  - numpy.ndarray: Serie progresiva de valores.
  """
  if num_steps <= 1:
      return np.array([start_value])

  x = np.linspace(0, 1, num_steps)
  # Aplicar aceleración
  progression = start_value + (end_value - start_value) * (x ** acceleration)
  return progression

# ### Función para descargar datos de stock con caché
# ### Función para descargar datos de stock con caché
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Handle MultiIndex columns: extract 'Close' for the ticker
        if isinstance(df.columns, pd.MultiIndex):
            # Check if ticker exists in the columns
            if ('Close', ticker) in df.columns:
                return df[('Close', ticker)]
            else:
                st.error(f"No 'Close' data found for ticker {ticker}")
                return None
        else:
            # Fallback for non-MultiIndex (unlikely with current yfinance)
            if 'Close' in df.columns:
                return df['Close']
            else:
                st.error(f"No 'Close' data found for ticker {ticker}")
                return None
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return None
# Descargar datos de CEDEAR y activo subyacente
with st.spinner("Descargando datos de CEDEAR..."):
  stock_data = get_stock_data(ticker, start_date, end_date)

with st.spinner("Descargando datos del activo subyacente..."):
  underlying_data = get_stock_data(underlying_ticker, start_date, end_date)

# Validación de datos descargados
if stock_data is None:
  st.error(f"No se encontraron datos para el ticker CEDEAR '{ticker}'. Verifique el símbolo y el rango de fechas.")
  st.stop()

if underlying_data is None:
  st.error(f"No se encontraron datos para el activo subyacente '{underlying_ticker}'. Verifique el símbolo y el rango de fechas.")
  st.stop()

# ### Ajuste de datos de stock por splits
try:
  adjusted_stock_data = ajustar_precios_por_splits(stock_data, ticker)
except Exception as e:
  st.error(f"Error al ajustar los precios por splits: {e}")
  st.stop()

# ### Obtención del ratio de conversión de CEDEAR
ticker_lookup = ticker.replace('.BA', '')

if ticker_lookup in cedear_ratios.index:
  try:
      conversion_ratio = float(cedear_ratios.loc[ticker_lookup, 'Ratio'])
  except Exception as e:
      st.error(f"Error al obtener el ratio para {ticker_lookup}: {e}")
      st.stop()
else:
  st.error(f"No se encontró un ratio para {ticker_lookup}.")
  st.stop()

# ### Obtener la tasa de cambio actual usando YPFD.BA y YPF
with st.spinner("Calculando la tasa de cambio actual..."):
  ypf_ba_data = get_stock_data('YPFD.BA', start_date, end_date)  # Ticker corregido
  ypf_data = get_stock_data('YPF', start_date, end_date)

  if ypf_ba_data is None or ypf_data is None:
      st.error("No se encontraron datos para YPFD.BA o YPF en el rango de fechas seleccionado.")
      st.stop()

  # Asegurar que hay fechas comunes para calcular la tasa de cambio
  common_dates = ypf_ba_data.index.intersection(ypf_data.index)
  if common_dates.empty:
      st.error("No hay fechas coincidentes entre YPFD.BA y YPF para calcular la tasa de cambio.")
      st.stop()

  # Calcular la tasa de cambio actual
  current_exchange_rate = ypf_ba_data.loc[common_dates].iloc[-1] / ypf_data.loc[common_dates].iloc[-1]

# ### Función para calcular tasa de cambio proyectada con eventos de usuario y progresión
def compute_projected_exchange_rates(start_date, end_date, current_rate, start_rate, end_rate, exchange_events, acceleration=1.0):
  """
  Calcular la serie de tasas de cambio USD/ARS proyectadas, incorporando eventos definidos por el usuario
  y una progresión no lineal de start_rate a end_rate.

  Parámetros:
  - start_date (pd.Timestamp): Inicio del período de proyección.
  - end_date (pd.Timestamp): Fin del período de proyección.
  - current_rate (float): Tasa de cambio actual.
  - start_rate (float): Tasa de cambio inicial en la proyección.
  - end_rate (float): Tasa de cambio final en la proyección.
  - exchange_events (list of dict): Lista de {'Fecha': pd.Timestamp, 'USD/ARS': rate}
  - acceleration (float): Factor de aceleración para la progresión.

  Retorna:
  - pd.Series: Tasas de cambio diarias proyectadas desde start_date hasta end_date.
  """
  projection_dates = pd.date_range(start=start_date, end=end_date, freq='D')
  num_days = len(projection_dates)
  
  # Crear progresión no lineal para la tasa de cambio base
  base_rates = create_progressive_series(start_rate, end_rate, num_days, acceleration=acceleration)
  exchange_rate_series = pd.Series(base_rates, index=projection_dates)

  # Incorporar eventos de tipo de cambio
  for event in exchange_events:
      event_date = pd.Timestamp(event["Fecha"])
      event_rate = event["USD/ARS"]
      if start_date <= event_date <= end_date:
          # Asegurarse de que la fecha del evento esté dentro del rango de proyección
          if event_date in exchange_rate_series.index:
              exchange_rate_series.loc[event_date] = event_rate
          else:
              # Si por alguna razón el event_date no está en el índice, agregarlo
              exchange_rate_series = exchange_rate_series.append(pd.Series([event_rate], index=[event_date]))
  
  # Reordenar el índice después de posibles inserciones
  exchange_rate_series = exchange_rate_series.sort_index()
  
  # Interpolar para llenar cualquier valor faltante debido a la inserción de eventos
  exchange_rate_series = exchange_rate_series.interpolate(method='quadratic')
  
  return exchange_rate_series

# ### Determinar la Fecha de Inicio de la Proyección
# Determinar la última fecha disponible en los datos históricos para iniciar la proyección
last_historical_date = adjusted_stock_data.index.max()

# Asignar projection_start_date como el día siguiente de la última fecha histórica
projection_start_date = last_historical_date + pd.Timedelta(days=1)

# Asegurarse de que projection_start_date no sea posterior a end_date
if projection_start_date >= end_date_ts:
  st.error("La fecha de inicio de la proyección está después de la fecha de finalización.")
  st.stop()

# ### Calcular las tasas de cambio proyectadas con progresión y aceleración
projected_exchange_rates = compute_projected_exchange_rates(
  start_date=projection_start_date,
  end_date=end_date_ts,
  current_rate=current_exchange_rate,
  start_rate=start_exchange_rate,
  end_rate=end_exchange_rate,
  exchange_events=exchange_events,  # Lista de diccionarios
  acceleration=exchange_acceleration  # Nuevo parámetro
)

# ### Verificación de Eventos de Tipo de Cambio Incluidos
st.subheader("Eventos de Tipo de Cambio Incluidos en la Proyección")

included_events = [
  event for event in exchange_events
  if projection_start_date <= pd.Timestamp(event["Fecha"]) <= end_date_ts
]

if included_events:
  events_df = pd.DataFrame(included_events)
  events_df['Fecha'] = pd.to_datetime(events_df['Fecha'])
  st.table(events_df)
else:
  st.write("No hay eventos de tipo de cambio incluidos en el período de proyección.")

# ### Gráfico de Tasas de Cambio Proyectadas
st.subheader("Tasas de Cambio Proyectadas (USD/ARS)")

exchange_rate_fig = go.Figure()

exchange_rate_fig.add_trace(go.Scatter(
  x=projected_exchange_rates.index,
  y=projected_exchange_rates,
  mode='lines',
  name='USD/ARS Proyectado',
  line=dict(color='purple')
))

# Marcar los eventos de tipo de cambio definidos por el usuario
for event in included_events:
  exchange_rate_fig.add_shape(
      type="line",
      x0=event["Fecha"],
      y0=projected_exchange_rates.min(),
      x1=event["Fecha"],
      y1=projected_exchange_rates.max(),
      line=dict(color="Red", dash="dash"),
  )
  exchange_rate_fig.add_annotation(
      x=event["Fecha"],
      y=projected_exchange_rates[event["Fecha"]],
      text=f"{event['USD/ARS']}",
      showarrow=True,
      arrowhead=1,
      yanchor="bottom"
  )

exchange_rate_fig.update_layout(
  title="Proyección de Tasas de Cambio USD/ARS",
  xaxis_title="Fecha",
  yaxis_title="USD/ARS",
  legend_title="Leyenda",
  hovermode='x unified'
)

st.plotly_chart(exchange_rate_fig, use_container_width=True)

# ### Proyecciones Futuras: Precio ajustado por inflación y desempeño esperado
predicted_dates = pd.date_range(start=projection_start_date, end=end_date_ts, freq='D')
num_days = len(predicted_dates)

if num_days == 0:
  st.error("El período de proyección no contiene días.")
  st.stop()

# Asegurar que projected_exchange_rates cubre todas las fechas de predicción
projected_exchange_rates = projected_exchange_rates.reindex(predicted_dates, method='nearest', fill_value=current_exchange_rate)

# Precios iniciales para iniciar la proyección
initial_underlying_price = underlying_data.iloc[-1]  # Último precio disponible del activo subyacente
initial_ce_dear_price = adjusted_stock_data.iloc[-1]  # Último precio ajustado del CEDEAR

# **Overhaul: Calcular Progresión de Inflación con Aceleración**

# Generar una serie progresiva de tasas de inflación diarias con aceleración
inflation_monthly_progression = create_progressive_series(
  start_inflation_rate,
  end_inflation_rate,
  num_days,
  acceleration=inflation_acceleration
)
inflation_daily_rates = (1 + inflation_monthly_progression) ** (1 / 30) - 1  # Convertir a tasa diaria

# Calcular la inflación acumulada progresiva
inflacion_following_prices = pd.Series(initial_ce_dear_price * (1 + inflation_daily_rates).cumprod(), index=predicted_dates)

# **Proyección basada en desempeño esperado**

# La tasa de crecimiento del activo subyacente ya se define como una tasa diaria
# Crear una progresión no lineal para la tasa de crecimiento del activo subyacente si es necesario
# En este caso, mantendremos la tasa diaria constante
daily_growth_rate = (1 + growth_rate_underlying_asset) ** (1 / 365) - 1
future_prices_underlying = initial_underlying_price * (1 + daily_growth_rate) ** np.arange(num_days)

# Calcular el precio esperado basado en el crecimiento del activo subyacente y tasas de cambio proyectadas
expected_performance_prices = (future_prices_underlying / conversion_ratio) * projected_exchange_rates.values

# Crear DataFrame para las proyecciones
projection_df = pd.DataFrame({
  'Date': predicted_dates,
  'CEDEAR Sigue Inflación': inflacion_following_prices,  # Removed .values
  'CEDEAR Esperado': expected_performance_prices
})

# Calcular el rendimiento porcentual relativo al precio inicial
projection_df['CEDEAR Sigue Inflación (%)'] = (projection_df['CEDEAR Sigue Inflación'] / initial_ce_dear_price - 1) * 100
projection_df['CEDEAR Esperado (%)'] = (projection_df['CEDEAR Esperado'] / initial_ce_dear_price - 1) * 100

# ### Graficar Proyecciones (Precio Absoluto)
fig = go.Figure()

# Precio histórico ajustado por splits
fig.add_trace(go.Scatter(
  x=adjusted_stock_data.index,
  y=adjusted_stock_data,
  mode='lines',
  name='Precio Histórico (Ajustado por Splits)',
  line=dict(color='blue')
))

# Proyección siguiendo la inflación
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Sigue Inflación'],
  mode='lines',
  name='CEDEAR Sigue Inflación',
  line=dict(color='green', dash='dash')
))

# Proyección basada en desempeño esperado
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Esperado'],
  mode='lines',
  name='CEDEAR Esperado',
  line=dict(color='orange', dash='dot')
))

# Ajustes de layout
fig.update_layout(
  title="Proyección del rendimiento de CEDEAR ajustado por inflación",
  xaxis_title="Fecha",
  yaxis_title="Precio Ajustado (ARS)",
  legend_title="Leyenda",
  hovermode='x unified'
)

# Mostrar el primer gráfico interactivo
st.plotly_chart(fig, use_container_width=True)

# ### Graficar Proyecciones (Rendimiento Porcentual)
fig2 = go.Figure()

# Rendimiento siguiendo la inflación (%)
fig2.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Sigue Inflación (%)'],
  mode='lines',
  name='CEDEAR Sigue Inflación (%)',
  line=dict(color='green', dash='dash')
))

# Rendimiento basado en desempeño esperado (%)
fig2.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Esperado (%)'],
  mode='lines',
  name='CEDEAR Esperado (%)',
  line=dict(color='orange', dash='dot')
))

# Ajustes de layout para el segundo gráfico
fig2.update_layout(
  title="Proyección del rendimiento de CEDEAR en Porcentajes",
  xaxis_title="Fecha",
  yaxis_title="Rendimiento (%)",
  legend_title="Leyenda",
  hovermode='x unified'
)

# Mostrar el segundo gráfico interactivo
st.plotly_chart(fig2, use_container_width=True)

# ### Resumen de la Proyección
st.subheader("Resumen de Proyección")

# Precios proyectados finales
final_inflacion_price = inflacion_following_prices.iloc[-1]
final_expected_price = expected_performance_prices[-1]

# Cálculo de rendimiento porcentual
performance_inflacion = (final_inflacion_price / initial_ce_dear_price - 1) * 100
performance_expected = (final_expected_price / initial_ce_dear_price - 1) * 100

# Mostrar métricas para la proyección que sigue la inflación
st.write(f"**Precio inicial ajustado:** ${initial_ce_dear_price:,.2f} ARS")
st.write(f"**Precio proyectado ajustado al {end_date_ts.strftime('%d/%m/%Y')} (Sigue Inflación):** ${final_inflacion_price:,.2f} ARS")
st.write(f"**Rendimiento proyectado ajustado por inflación (Sigue Inflación):** {performance_inflacion:.2f}%")
st.write("---")  # Línea horizontal para separación

# Mostrar métricas para la proyección basada en desempeño esperado
st.write(f"**Precio proyectado ajustado al {end_date_ts.strftime('%d/%m/%Y')} (Esperado):** ${final_expected_price:,.2f} ARS")
st.write(f"**Rendimiento proyectado ajustado por inflación (Esperado):** {performance_expected:.2f}%")

# ### Tabla de Comparación de Proyecciones
st.subheader("Comparación de Proyecciones")

comparison_df = pd.DataFrame({
  'Proyección': ['Sigue Inflación', 'Esperado'],
  'Precio Proyectado (ARS)': [final_inflacion_price, final_expected_price],
  'Rendimiento Proyectado (%)': [performance_inflacion, performance_expected]
})

st.table(comparison_df)
