import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import logging
import re
import requests
import urllib3
import time  # Added missing import
import os
from curl_cffi import requests as cffi_requests
from retrying import retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurar logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO for better debugging
logger = logging.getLogger(__name__)

# ------------------------------
# Diccionario de tickers y sus divisores
splits = {
    'ADGO.BA': 1,
    'ADBE.BA': 2,
    'AEM.BA': 2,
    'AMGN.BA': 3,
    'AAPL.BA': 2,
    'BAC.BA': 2,
    'GOLD.BA': 2,
    'BIOX.BA': 2,
    'CVX.BA': 2,
    'LLY.BA': 7,
    'XOM.BA': 2,
    'FSLR.BA': 6,
    'IBM.BA': 3,
    'JD.BA': 2,
    'JPM.BA': 3,
    'MELI.BA': 2,
    'NFLX.BA': 3,
    'PEP.BA': 3,
    'PFE.BA': 2,
    'PG.BA': 3,
    'RIO.BA': 2,
    'SONY.BA': 2,
    'SBUX.BA': 3,
    'TXR.BA': 2,
    'BA.BA': 4,
    'TM.BA': 3,
    'VZ.BA': 2,
    'VIST.BA': 3,
    'WMT.BA': 3,
    'AGRO.BA': (6, 2.1)
}

# ------------------------------
# Data source functions
@retry(stop_max_attempt_number=3, wait_fixed=5000)  # Retry 3 times, wait 5 seconds
def descargar_datos_yfinance(ticker, start, end):
    try:
        # Cache file path
        cache_file = f"cache/{ticker}_{start}_{end}.csv"
        os.makedirs("cache", exist_ok=True)

        # Check if cached data exists
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            logger.info(f"Datos cargados desde caché para {ticker}")
            return df

        # Create a curl_cffi session with updated Chrome impersonation
        session = cffi_requests.Session(impersonate="chrome131")  # Updated to chrome131

        # Add headers for additional browser-like behavior
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Download data using yfinance with the custom session
        stock_data = yf.download(ticker, start=start, end=end, progress=False, session=session)

        if stock_data.empty:
            logger.warning(f"No se encontraron datos para el ticker {ticker} en el rango de fechas seleccionado.")
            return pd.DataFrame()

        # Extract just the Close column and handle MultiIndex safely
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Check if 'Close' level exists
            if 'Close' in stock_data.columns.levels[0]:
                close = stock_data['Close']
                # If ticker is in the second level, use it; otherwise, assume single ticker
                if ticker in close.columns:
                    close = close[ticker].to_frame('Close')
                else:
                    close = close.iloc[:, 0].to_frame('Close')  # Fallback to first column
            else:
                logger.error(f"No 'Close' column found for {ticker} in MultiIndex.")
                return pd.DataFrame()
        else:
            if 'Close' in stock_data.columns:
                close = stock_data[['Close']]
            else:
                logger.error(f"No 'Close' column found for {ticker}.")
                return pd.DataFrame()

        # Save to cache
        close.to_csv(cache_file)
        logger.info(f"Datos guardados en caché para {ticker}")

        return close

    except Exception as e:
        logger.error(f"Error downloading data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from analisistecnico for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                logger.error(f"Error in API response for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['bars'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df['Close'] = df['close']
            df = df[['Date', 'Close']]
            df.set_index('Date', inplace=True)
            df = df.sort_index().drop_duplicates()
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from IOL for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }

        symbol = ticker.replace('.BA', '') + ' 24HS'

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from ByMA Data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def descargar_datos(ticker, start, end, source='YFinance'):
    try:
        if source == 'YFinance':
            df = descargar_datos_yfinance(ticker, start, end)
            # Fallback for .BA tickers if yfinance fails
            if df.empty and ticker.endswith('.BA'):
                logger.warning(f"yfinance falló para {ticker}, intentando con analisistecnico...")
                df = descargar_datos_analisistecnico(ticker, start, end)
                if df.empty:
                    logger.warning(f"analisistecnico falló para {ticker}, intentando con iol...")
                    df = descargar_datos_iol(ticker, start, end)
                    if df.empty:
                        logger.warning(f"iol falló para {ticker}, intentando con byma...")
                        df = descargar_datos_byma(ticker, start, end)
        elif source == 'AnálisisTécnico.com.ar':
            df = descargar_datos_analisistecnico(ticker, start, end)
        elif source == 'IOL (Invertir Online)':
            df = descargar_datos_iol(ticker, start, end)
        elif source == 'ByMA Data':
            df = descargar_datos_byma(ticker, start, end)
        else:
            logger.error(f"Unknown data source: {source}")
            return pd.DataFrame()
        time.sleep(5)  # Increased to 5 seconds to reduce rate-limiting
        return df
    except Exception as e:
        logger.error(f"Error downloading data for {ticker} from {source}: {e}")
        return pd.DataFrame()

# ------------------------------
# Helper functions
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker in splits:
            adjustment = splits[ticker]
            if isinstance(adjustment, tuple):
                split_date = datetime(2023, 11, 3)
                # Create separate DataFrames for different date ranges
                df_before_split = df[df.index < split_date].copy()
                df_on_split = df[df.index == split_date].copy()
                df_after_split = df[df.index > split_date].copy()
                
                # Adjust prices before split date
                df_before_split['Close'] /= adjustment[0]  # Divide by 6 for AGRO.BA
                # Adjust prices on split date
                df_on_split['Close'] *= adjustment[1]     # Multiply by 2.1 for AGRO.BA
                # Prices after split date remain unchanged
                
                # Concatenate and sort
                df = pd.concat([df_before_split, df_on_split, df_after_split]).sort_index()
            else:
                split_threshold_date = datetime(2024, 1, 23)
                df.loc[df.index <= split_threshold_date, 'Close'] /= adjustment
    except Exception as e:
        logger.error(f"Error ajustando splits para {ticker}: {e}")
    return df

@st.cache_data
def load_cpi_data():
    try:
        cpi = pd.read_csv('inflaciónargentina2.csv')
        if 'Date' not in cpi.columns or 'CPI_MoM' not in cpi.columns:
            st.error("El archivo CSV debe contener las columnas 'Date' y 'CPI_MoM'.")
            st.stop()

        cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
        cpi.set_index('Date', inplace=True)
        cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
        daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')

        # Ensure index is datetime without timezone
        daily.index = pd.to_datetime(daily.index)
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        return daily
    except FileNotFoundError:
        st.error("El archivo 'inflaciónargentina2.csv' no se encontró.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        st.stop()

# Load CPI data
daily_cpi = load_cpi_data()

# ------------------------------
# Streamlit UI
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - [X: MTaurus_ok](https://x.com/MTaurus_ok)')

# Add data source selector in sidebar
st.sidebar.title("Configuración")
data_source = st.sidebar.radio(
    "Fuente de datos:",
    ('YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data')
)

# Add information about data sources
st.sidebar.markdown("""
### Información sobre fuentes de datos:
- **YFinance**: Datos internacionales, mejor para tickers extranjeros
- **AnálisisTécnico.com.ar**: Datos locales, mejor para tickers argentinos
- **IOL**: Datos locales con acceso a bonos y otros instrumentos
- **ByMA Data**: Datos oficiales del mercado argentino

*Nota: Algunos tickers pueden no estar disponibles en todas las fuentes.*
""")

# ------------------------------
# Calculador de precios por inflación
st.subheader('1- Calculador de precios por inflación')

value_choice = st.radio(
    "¿Quieres ingresar el valor para la fecha de inicio o la fecha de fin?",
    ('Fecha de Inicio', 'Fecha de Fin'),
    key='value_choice_radio'
)

if value_choice == 'Fecha de Inicio':
    start_date = st.date_input(
        'Selecciona la fecha de inicio:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_input'
    )
    end_date = st.date_input(
        'Selecciona la fecha de fin:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_input'
    )
    start_value = st.number_input(
        'Ingresa el valor en la fecha de inicio (en ARS):',
        min_value=0.0,
        value=100.0,
        key='start_value_input'
    )

    try:
        start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
        end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
        end_value = start_value * (end_inflation / start_inflation)
        st.write(f"Valor inicial el {start_date}: ARS {start_value}")
        st.write(f"Valor ajustado el {end_date}: ARS {end_value:.2f}")
    except KeyError as e:
        st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
        st.stop()

else:
    start_date = st.date_input(
        'Selecciona la fecha de inicio:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_end_date_input'
    )
    end_date = st.date_input(
        'Selecciona la fecha de fin:',
        min_value=start_date,
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_end_date_input'
    )
    end_value = st.number_input(
        'Ingresa el valor en la fecha de fin (en ARS):',
        min_value=0.0,
        value=100.0,
        key='end_value_input'
    )

    try:
        start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
        end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
        start_value = end_value / (end_inflation / start_inflation)
        st.write(f"Valor ajustado el {start_date}: ARS {start_value:.2f}")
        st.write(f"Valor final el {end_date}: ARS {end_value}")
    except KeyError as e:
        st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
        st.stop()

# ------------------------------
# Ajustadora de acciones por inflación
st.subheader('2- Ajustadora de acciones por inflación')

tickers_input = st.text_input(
    'Ingresa los tickers de acciones separados por comas (por ejemplo, AAPL.BA, MSFT.BA, META):',
    key='tickers_input'
)

sma_period = st.number_input(
    'Ingresa el número de periodos para el SMA del primer ticker:',
    min_value=1,
    value=10,
    key='sma_period_input'
)

plot_start_date = st.date_input(
    'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=(daily_cpi.index.max() - timedelta(days=365)).date(),
    key='plot_start_date_input'
)

show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False)
show_percentage_from_recent = st.checkbox(
    'Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente',
    value=False
)

# Diccionarios para almacenar datos
stock_data_dict_nominal = {}
stock_data_dict_adjusted = {}

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    fig = go.Figure()
    ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}

    for i, ticker in enumerate(tickers):
        try:
            # Download data using selected source
            stock_data = descargar_datos(ticker, plot_start_date, daily_cpi.index.max().date(), data_source)

            if stock_data.empty:
                st.error(f"No se encontraron datos para el ticker {ticker}.")
                continue

            # Ensure datetime index and remove any timezone information
            if 'Date' in stock_data.columns:
                stock_data.set_index('Date', inplace=True)
            stock_data.index = pd.to_datetime(stock_data.index)
            if stock_data.index.tz is not None:
                stock_data.index = stock_data.index.tz_localize(None)

            # For IOL and ByMA, rename the price column to 'Close'
            if data_source in ['IOL (Invertir Online)', 'ByMA Data']:
                if len(stock_data.columns) == 1:
                    stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Close'})

            # Fix timezone offset in index
            stock_data.index = stock_data.index.tz_localize(None)
            stock_data.index = stock_data.index.normalize()  # Remove time component

            # Apply splits adjustment
            stock_data = ajustar_precios_por_splits(stock_data, ticker)

            # Determine if inflation adjustment is needed based on data source and
