
import backtrader as bt
from numpy import arange
from math import log, exp, log2, sqrt
from scipy.stats import norm

n = norm.pdf
N = norm.cdf

class TestStrategy(bt.Strategy): 
    def __init__(self) -> None:
        self.buyPrice = None
        self.salePrice = None
        self.order = None
        pass
    def log(self):
        pass
    def notify_order(self):
         pass
    def next(self):
        self.buy()
        self.log()
        pass
    def bs_price(cp_flag,S,K,T,r,v,q=0.0):
        d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
        d2 = d1-v*sqrt(T)
        if cp_flag == 'c':
            price = S*exp(-q*T)*N(d1)-K*exp(-r*T)*N(d2)
        else:
            price = K*exp(-r*T)*N(-d2)-S*exp(-q*T)*N(-d1)
        return price
    def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
        d1 = (log2(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
        return S * sqrt(T)*n(d1)
    def find_vol(self, target_value, call_put, S, K, T, r):
        MAX_ITERATIONS = 100
        PRECISION = 1.0e-5
        sigma = 0.5
        for i in arange(0, MAX_ITERATIONS):
            price = self.bs_price(call_put, S, K, T, r, sigma)
            vega = self.bs_vega(call_put, S, K, T, r, sigma)
            price = price
            diff = target_value - price  # 我们的根
            print(i, sigma, diff)
            if (abs(diff) < PRECISION):
                return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)

        # 未找到值，返回迄今为止的最佳猜测
        return sigma


import pandas as pd  
import numpy as np  
  
# 读取期权数据  
# 假设数据包括日期、期权价格、期权行权价、期权到期时间、标的资产价格等信息  
data = pd.read_csv('option_data.csv')  
  
# 定义策略参数  
# 假设策略为买入期权，当标的资产价格高于行权价时执行买入操作  
# 假设买入期权的到期时间为3个月  
entry_threshold = 105  # 买入期权的行权价  
exit_threshold = 100  # 卖出期权的行权价  
expiration_time = 3  # 期权到期时间（单位：个月）  
  
# 计算买入信号  
# 当标的资产价格高于买入期权的行权价时，发出买入信号  
data['buy_signal'] = np.where(data['underlying_price'] > entry_threshold, 1, 0)  
  
# 计算卖出信号  
# 当买入期权的到期时间到达时，发出卖出信号  
data['sell_signal'] = np.where((data['expiration_time'] - data['expiration_time'].shift(1)) == expiration_time, 1, 0)  
  
# 计算持仓方向和持仓数量  
# 在有买入信号时开仓，在有卖出信号时平仓  
data['position'] = data['buy_signal'] - data['sell_signal']  
  
# 计算策略收益  
# 假设期权收益为行权价与标的资产价格之差的绝对值  
data['profit'] = np.abs(data['underlying_price'] - data['strike_price']) * data['position']  
  
# 输出策略收益  
print(data['profit'].sum())

import numpy as np
from scipy import stats
from scipy.optimize import bisect

# 定义期权定价模型（这里使用Black-Scholes模型）
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

# 定义隐含波动率的目标函数
def implied_volatility(option_price, S, K, T, r, target_price):
    return lambda sigma: black_scholes_call(S, K, T, r, sigma) - target_price

# 输入期权参数
S = 100  # 标的资产价格
K = 100  # 期权行权价格
T = 1.0  # 期权到期时间（以年为单位）
r = 0.05  # 无风险利率
option_price = 10  # 市场观察到的期权价格

# 使用二分法求解隐含波动率
iv_func = implied_volatility(option_price, S, K, T, r, option_price)
implied_volatility = bisect(iv_func, 0.01, 2.0)  # 初始波动率范围取0.01到2.0

print(f"隐含波动率为: {implied_volatility:.4f}")
    
