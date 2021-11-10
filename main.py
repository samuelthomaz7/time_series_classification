from TSFE import TSFE
import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.seasonal as sea
from statsmodels.tsa.stattools import acf
import statsmodels.graphics.tsaplots as sgt
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf
from scipy.stats import chi2
from scipy.stats import f
import matplotlib.pyplot as plt
# from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
# from scipy.stats import pearsonr
import pmdarima
from math import log

def prepare_dataset(df, column_date):
    
    df[column_date] = pd.to_datetime(df[column_date])
    df.index = df[column_date]
    df.index.name = ''
    idx = pd.date_range(start = df[column_date].min(),end =  df[column_date].max())

    df = df.reindex(idx, fill_value = np.nan)

    df = df.fillna(method='bfill')
    
    return df
    
    

# flights = prepare_data_set(df=flights, column_date='Month')


temperature = pd.read_csv('min_temp.csv')
flights = pd.read_csv('AirPassengers.csv')
births = pd.read_csv('female_birth.csv')
# pedestres = pd.read_csv('C:/Users/samue/Desktop/UCRArchive_2018/Chinatown/Chinatown_TEST.tsv', sep = '\t', encoding='utf-8')
sunspot = sm.datasets.sunspots.data.load_pandas().endog
lynx = pmdarima.datasets.load_lynx(as_series=True)

births = prepare_dataset(df = births, column_date='Date')
temperature = prepare_dataset(df = temperature, column_date='Date')

flights['Month'] = pd.to_datetime(flights['Month'])
flights.index = flights['Month']
flights.index.name = ''

from yahoo_fin.stock_info import get_data
amazon= get_data("amzn", start_date="12/01/2018", end_date="12/04/2019", index_as_date = True, interval="1d")

class_test = TSFE()
# class_test.get_all_features(x = flights['#Passengers'])  

datasets = {
    'lynx':lynx,
    'sunspot': sunspot, 
    'flights': flights['#Passengers'],
    "births":births.Births, 
    'amazon':amazon.close,
    # 'temp':temperature.Temp
}

for name, x in datasets.items():
    print(name, class_test.get_all_features(x = x))

print("asd")
