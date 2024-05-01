import pandas as pd
import numpy as np
import FinanceDataReader as fdr

from concurrent.futures import ProcessPoolExecutor
import os


def nz(value, default):
    if value is None or np.isnan(value):
        return default
    return value

class BBand_Strategy():
    def __init__(self, df):
        self.df = df
        self.close = self.df['Close']
        self.open = self.df['Open']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.volume = self.df['Volume']     


    def isUpperBBand(self, length = 15, stdev = 2):
        minwidth = 3
        diff = True
        diffperc = 0

        #Calculate and plot the Bollinger Bands
        lbb = pd.DataFrame()

        lbb['lbbmiddle'] = self.close.rolling(window=length).mean()
        lbb['lbbdev'] = stdev * self.close.rolling(window=length).std()
        lbb['lbbdev'] = np.maximum(lbb['lbbdev'], (lbb['lbbmiddle'] * minwidth) / 100)

        # lbb['lbbdev'] = max(lbb['lbbdev'], (lbb['lbbmiddle'] * minwidth) / 100)
        lbbupper = lbb['lbbmiddle'] + lbb['lbbdev']
        lbblower = lbb['lbbmiddle'] - lbb['lbbdev']

        lbb['lbbcrossover'] = (self.close >= lbbupper) & (self.close.shift(1) < lbbupper.shift(1))
        lbbcrossover = lbb['lbbcrossover']
        lbbdiffcond = ((lbbupper / lbblower) * 100) - 100 >= diffperc if diff else True
       
        return lbbcrossover & lbbdiffcond
        


    # def arma(self, length=13, gamma=3):
    #     enable = False
    #     ma = np.zeros(len(self.df))
    #     mad = np.zeros(len(self.df))
    #     src_ = self.close
    #     ma = nz(mad, src_)
    #     bar_index = np.arange(1, len(src_) + 1)
    #     d = (np.abs(src_.iloc[length] - ma)).cumsum() / bar_index * gamma

    #     # d = (np.abs(src_[length] - ma)).cumsum() / bar_index * gamma
    #     condition1 = src_ > nz(mad[-2], src_) + d
    #     condition2 = src_ < nz(mad[-2], src_)
    #     mad = np.where(condition1, src_ + d, np.where(condition2, src_ - d, src_))
    #     mad = pd.Series(mad).rolling(window=length).mean().rolling(window=length).mean()

    #     # d = (np.abs(src_[length] - ma)).cumsum() / bar_index * gamma
    #     # mad = ((src_ + d if src_ > nz(mad[-2], src_) + d else (src_ - d if src_ < nz(mad[-2], src_) else src_)).rolling(window = length).mean()).rolling(window = length).mean()
        
    #     return self.close.iloc[-1] > mad.iloc[-1]


    def arma(self, length=13, gamma=2):
        src_ = self.close
        src_ = src_.reset_index(drop=True)
        ma = pd.Series(np.zeros(len(src_),dtype=float),index=src_.index)
        mad = pd.Series(np.zeros(len(src_),dtype=float),index=src_.index)
        
        bar_index = pd.Series(np.arange(1, len(src_) + 1))
        
        # nz 함수와 같은 기능을 수행
        ma = mad.shift(1).fillna(src_)

        # ta.cum과 math.abs, 그리고 인덱싱 대응
        d = (np.abs(src_.shift(length) - ma).cumsum() / bar_index) * gamma

        # 조건부 할당 대응
        mad = np.where(src_ > mad.shift(1).fillna(src_) + d, src_ + d, np.where(src_ < mad.shift(1).fillna(src_) - d, src_ - d, mad.shift(1).fillna(src_)))
        mad = pd.Series(mad, index=src_.index)  # np.where 결과를 pandas Series로 변환

        # 이동 평균 계산
        mad = pd.Series(mad).rolling(window=length).mean().rolling(window=length).mean()

        return src_ > mad

        
    def volatility(self, std_length = 3, ma_length = 36):
        vola_std = self.close.rolling(window = std_length).std()
        vola_ma = vola_std.rolling(window = ma_length).mean()
    
        
        return vola_std > vola_ma


    def history_vola(self, HVlength = 6, HVthreshold = 1):
        HV = np.log(self.close / self.close.shift(1))
        HV = HV.rolling(HVlength).std()
        HV = HV * np.sqrt(365) * 100

        return (HV - HV.shift(1)) > HVthreshold
        
    ### 여기부터 만들어야 할 차례
    ### 근데 사실 이거만 하면 끝인데 코드를 좀 단순화 시키고 싶어
        
        
    def adx(self, adxlen = 17):
        
        tr = np.maximum.reduce([
            self.high.astype(float) - self.low.astype(float), 
            np.abs(self.high.astype(float) - np.roll(self.close.astype(float), 1)), 
            np.abs(self.low.astype(float) - np.roll(self.close.astype(float), 1))
            ])
        tr[0] = np.nan  # 첫 번째 값은 비교 대상이 없으므로 NaN 처리

        # tr = np.maximum.reduce([self.high - self.low, np.abs(self.high - np.roll(self.close, 1)), np.abs(self.low - np.roll(self.close, 1))])
        # tr[0] = np.nan  # 첫 번째 값은 비교 대상이 없으므로 NaN 처리
        
        dm_plus = np.where(self.high - np.roll(self.high, 1) > np.roll(self.low, 1) - self.low, np.maximum(self.high - np.roll(self.high, 1), 0), 0)
        dm_minus = np.where(np.roll(self.low, 1) - self.low > self.high - np.roll(self.high, 1), np.maximum(np.roll(self.low, 1) - self.low, 0), 0)
        
        smoothed_tr = np.zeros_like(tr)
        smoothed_dm_plus = np.zeros_like(dm_plus)
        smoothed_dm_minus = np.zeros_like(dm_minus)
        
        for i in range(1, len(tr)):
            smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / adxlen) + tr[i]
            smoothed_dm_plus[i] = smoothed_dm_plus[i-1] - (smoothed_dm_plus[i-1] / adxlen) + dm_plus[i]
            smoothed_dm_minus[i] = smoothed_dm_minus[i-1] - (smoothed_dm_minus[i-1] / adxlen) + dm_minus[i]
        
        # di_plus_1 = smoothed_dm_plus[-2] / smoothed_tr[-2] * 100
        # di_plus = smoothed_dm_plus[-1] / smoothed_tr[-1] * 100
        # di_minus = smoothed_dm_minus[-1] / smoothed_tr[-1] * 100 
        epsilon = 1e-10

        # smoothed_tr이 0인 위치에서는 epsilon을 사용하고, 그렇지 않은 경우 smoothed_tr 사용
        safe_smoothed_tr = np.where(smoothed_tr == 0, epsilon, smoothed_tr)
        
        di_plus = smoothed_dm_plus / safe_smoothed_tr * 100
        di_minus = smoothed_dm_minus / safe_smoothed_tr * 100
        #dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100

        di = pd.DataFrame({'di_plus': di_plus, 'di_minus': di_minus})
        
        # adx = np.zeros_like(dx)
        # for i in range(adxlen, len(dx)):
        #     adx[i] = np.mean(dx[i-adxlen+1:i+1])

        return (di['di_plus'].shift(1) < di['di_plus']) & (di['di_plus'] > di['di_minus'])
        #return (di_plus[-2] < di_plus[-1]) & (di_plus[-1] > di_minus[-1])

    def check_all_conditions(self):
        conditions = [
            self.isUpperBBand(15, 2.3).iloc[-1],
            self.arma(13, 2).iloc[-1],
            self.volatility(3, 36).iloc[-1],
            self.history_vola(10, 1).iloc[-1],
            self.adx(18).iloc[-1]
        ]
        if all(conditions):
            return True
        else:
            return False

        # if all(conditions):
        #     print("모든 조건이 만족되었습니다: True")
        # else:
        #     print("모든 조건이 만족되지 않았습니다: False")


def read_ticker_file(filename):
    with open(filename, "r") as f:
        codes = [line.strip() for line in f.readlines()]
    return codes

filename = os.path.abspath("kosdaq.txt")
    
codes = read_ticker_file(filename)
codelist = []

# if __name__ == "__main__":
#     df = fdr.DataReader('005690','2024-01-01','2024-03-11')
#     bband = BBand_Strategy(df)
    
#     print(bband.check_all_conditions())

import time


def bbbo(code):
    df = fdr.DataReader(code, '2024-01-01', '2024-04-25')
    bband = BBand_Strategy(df)
    if bband.check_all_conditions():
        return code


if __name__ == '__main__':
    start_time = time.time()
    # max_workers = os.cpu_count() - 1
    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = [executor.submit(bbbo, code) for code in codes]
        results = [f.result() for f in futures]
    filtered_list = [item for item in results if item is not None]

    print(filtered_list)
    end_time = time.time()
    print(f"프로그램 실행 시간 : {end_time - start_time}초")





# if __name__ == '__main__':
    
#     start_time = time.time()
#     code_result = [bbbo(code) for code in codes]
#     print(code_result)
#     end_time = time.time()

#     print(f"프로그램 실행 시간 : {end_time - start_time}초")

