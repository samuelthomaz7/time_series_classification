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
from tqdm import tqdm
# import concurrent.futures
from joblib import Parallel, delayed


class TSFE():
    
    def __init__(self):
        pass
    
    
    def get_seasonality_and_trend(self, x):

        # doing the box-cox transformation
        complete_data = pd.DataFrame(x)

        x_array = x.values

        if(min(x_array) <= 0):
            x_array = x_array - min(x) + 1

        bc_obj =  boxcox(x_array)
        lmbda = bc_obj[1]
        transformed_series = bc_obj[0]

        complete_data['box_cox'] = transformed_series    


        period = self.get_periodicity(x_array, plot_bool = False)
    #     print(period)

        # calcule of seasonality strenght
        if(period != 1):   
            dec = seasonal_decompose(complete_data['box_cox'], period = period)
            seasonal_strength = 1 - np.var(dec.resid)/np.var(dec.resid + dec.seasonal)
        else:
            dec = seasonal_decompose(complete_data['box_cox'])
            seasonal_strength = 0

        # calcule of trend strenght

        trend_strength = 1 - np.var(dec.resid)/np.var(dec.resid + dec.trend)

        return {'trend': trend_strength, 'seasonal': seasonal_strength}
    

    def do_regression_spline(self, x, number_of_knots = 3):
        
        base_knot = (len(x))/(number_of_knots+1)
        knots = [int(i*base_knot) for i in range(1, number_of_knots + 1)]
        transformed_x = dmatrix(f"bs(train, knots={knots}, degree=3, include_intercept=False)", 
                            {"train": np.array(range(0,len(x)))},
                            return_type='dataframe')

        if(type(x) == type(pd.Series([1,2]))):  
            fit1 = sm.GLM(x.values, transformed_x).fit()
        else:
            fit1 = sm.GLM(x, transformed_x).fit()

        pred = fit1.predict(dmatrix(f"bs(valid, knots={knots}, include_intercept=False)", {"valid": np.array(range(0,len(x)))}, return_type='dataframe'))
        
            
        return pred

    def get_periodicity(self, x, number_of_knots = 3, plot_bool = True, figsize = (15,5)):
        
        pred1 = self.do_regression_spline(x = x, number_of_knots = number_of_knots)
        if(type(x) == type(pd.Series([1,2]))):  
            pred1.index = x.index

        
        
        auto_corr_df = pd.DataFrame()
        auto_corr_obj = acf(x - pred1, nlags=int(len(x)/3), alpha=0.05)
        
        auto_corr_df['acf'] = auto_corr_obj[0]
        
        if(int(len(auto_corr_df)/12) < 5):
            window = 5
        else:
            window = int(len(auto_corr_df)/12)
        
        
        auto_corr_df['moving_averge'] = auto_corr_df.rolling(window= window, center=True).mean()
        
        ma = auto_corr_df.moving_averge.values
        

        peaks, _= find_peaks(ma, distance=5)
        troughs, _= find_peaks(-ma, distance=5)
        
        if(plot_bool):
            
            rows = '1'
            columns = '2'
            
            
            plt.figure(figsize = figsize)
            plt.subplot(int(rows + columns + '1'))
            
            plt.plot(np.array(range(0,len(x))), x, label = 'Original')
            plt.plot(np.array(range(0,len(x))), pred1, label = f'Cubic spline, {number_of_knots} knots')
            plt.plot(np.array(range(0,len(x))), x -  pred1, label = 'Detrended')
            plt.legend()
            
            plt.subplot(int(rows + columns + '2'))
            plt.plot(ma)
            plt.plot(peaks,ma[peaks], '^')
            plt.plot(troughs,ma[troughs], 'v')
    #         plt.xlim([0,50])
            
            frequency = 1
            
    #         min(len(peaks), len(troughs))
            
    #     print(peaks)
    #     print(troughs)

        for index in range(0, min(len(peaks), len(troughs))):
            if(peaks[0] < troughs[0]):
    #             print(peaks[index + 1], troughs[index], peaks[index + 1] - troughs[index])
                
                acf_peak = auto_corr_df['acf'].iloc[peaks[index + 1]]
                acf_trough = auto_corr_df['acf'].iloc[troughs[index]]
                
                if((acf_peak - acf_trough) < 0.1):
                    continue
                else:
                    return peaks[index + 1]
                    print((acf_peak - acf_trough))
            else:
                acf_peak = auto_corr_df['acf'].iloc[peaks[index]]
                acf_trough = auto_corr_df['acf'].iloc[troughs[index]]
                
    #             print((acf_peak - acf_trough))
                
                if((acf_peak - acf_trough) > 0.1):
    #                 print('menor peaks', peaks[index], 'trohug', troughs[index], acf_peak - acf_trough)
                    return peaks[index]
                else:
                    pass
    #                 return peaks[index + 1]
    #                 print('maior peaks', peaks[index], 'trohug', troughs[index], acf_peak - acf_trough)
        
        
        
        return 1

    def get_skewness(self, x):
        
        n = len(x)
        sig = np.sqrt(np.var(x))
        mean = np.mean(x)
        
        cubic_sum = 0
        
        for xt in x:
            cubic_sum = cubic_sum + (xt - mean)**3
            
        skw = (1/(n*(sig**3)))*cubic_sum
        
        return skw


    def get_kurtosis(self, x):
        
        n = len(x)
        sig = np.sqrt(np.var(x))
        mean = np.mean(x)
        
        sum_4 = 0
        
        for xt in x:
            sum_4 = sum_4 + (xt - mean)**4
            
        skw = (1/(n*(sig**4)))*sum_4 - 3
        
        return skw

    def get_linearity(self, x):
        
        if(type(x) == type(pd.Series([1,2]))):
            used_x = x.values
        else:
            used_x = x
        
        period = self.get_periodicity(x = used_x, plot_bool=False)
        
        df = pd.DataFrame()
        df['x'] = x
        df['lag'] = df.x.shift(period)
        
        
        return df.corr().loc['x', 'lag']


    def get_serial_correlation(self, x, lags = 20, method = 'Box-Pierce', fitdf = 0):
        
        cor = acf(x = x, nlags=lags)
        if(type(x) == type(pd.Series([1,2]))):
            n = sum(~x.isna()) 
        else:
            n = sum(~np.isnan(np.array(x)))
        parameter = lags - fitdf
        obs = np.array(cor[1:])
    #     print()
        
        if(method == "Box-Pierce"):
            statistic = n * sum(obs**2)
            pval = 1 - chi2.cdf(x = statistic, df = lags - fitdf)
    #         pass

            return {'stat':statistic, 'pval':pval }


    def get_hurst_exponent(self, x, max_lag=20, plot = False):
        """Returns the Hurst Exponent of the time series"""
        
        if(type(x) == type(pd.Series([1,2]))):
            x = x.values 
        else:
            pass
        
        lags = range(2, max_lag)

        # variances of the lagged differences
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]

        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        if(plot):
            sns.scatterplot(x = np.log(lags), y = np.log(tau))
            plt.show()
        
        if(reg[0] < 0):
            return 0

        return reg[0]

    def get_lyapunov_exponent(self, data, plot = False, plot_pos = 0):
    
        N = len(data)            

        def get_list_of_mins(i):
            temp = []
            for j in range(1, N):
                temp.append(abs(data.iloc[i] - data.iloc[j]))

            return min(temp)

        list_of_mins = Parallel(n_jobs=8)(delayed(get_list_of_mins)(i) for i in tqdm(range(0,N)))

        data = data.values
        
        
        def getting_exps(i):
            lyapunovs = {}
            for j in range(i + 1, N):
                if np.abs(data[i] - data[j]) < list_of_mins[i] + 1:
                    lyapunovs[(i, j)] = []
                    for k in range(min(N - i, N - j)):
                        if(data[i+k] - data[j+k] != 0):
                            lyapunovs[(i, j)].append(log(np.abs(data[i+k] - data[j+k])))
                        else:
                            lyapunovs[(i, j)].append(1)
                            
            return lyapunovs
        
        list_range = [i for i in range(N)]
        lyapunovs = {}
        results = Parallel(n_jobs=8)(delayed(getting_exps)(i) for i in tqdm(range(N)))
        lyapunovs = {}

        for res in results:
            lyapunovs.update(res)
                            
        lya_exp = []
        for key, value in lyapunovs.items():
            df = pd.DataFrame()
            df['x'] = list(range(0,len(value)))
            df['y'] = value
            fmla = 'y ~ x'


            model = smf.ols(formula=fmla, data=df)
            res = model.fit()
            
            if plot:
                df['pred'] = [res.params['x']*i + res.params['Intercept'] for i in range(0,len(value))]
                print('exp', res.params['x'])
                df[['y', 'pred']].plot()
                plt.show()
                


                return None
        
            lya_exp.append(res.params['x'])
            
        return sum(lya_exp)/len(lyapunovs)
    
    
    def get_all_features(self, 
                         x,
                         trend_seasonality = True, 
                         skewness= True, 
                         kurtosis = True, 
                         linearity = True,
                         periodicity = True,
                         serial_correlation = True,
                         self_similarity = True,
                         chaos = True):
        
        features = {}
        
        if(trend_seasonality):
            features.update(self.get_seasonality_and_trend(x))
        
            
        if(skewness):
            features['skewness'] = self.get_skewness(x)
            
            
        if(kurtosis):
            features['kurtosis'] = self.get_kurtosis(x)
            
            
        if(periodicity):
            features['periodicity'] = self.get_periodicity(x = x, plot_bool=False)
            

        if(linearity):
            features['linearity'] = self.get_linearity(x = x.values)

        if(serial_correlation):
            features['serial_correlation'] = self.get_serial_correlation(x = x)['stat']

        if(self_similarity):
            features['self_similarity'] = self.get_hurst_exponent(x = x)

        if(chaos):
            features['chaos'] = self.get_lyapunov_exponent(data=x)
        


        return features
            



class_test = TSFE()
# class_test.get_all_features(x = flights['#Passengers'])               