import GPminer as GPm
import pandas as pd
import numpy as np
import FreeBack as FB


# 因子命名格式：
# factor-method-period
# factor: 基础因子名， 如果有参数，用.分割
# method：时序算符, period: 时序算符周期
class Factor():
    def __init__(self, market):
        for i in ['close', 'open', 'high', 'low', 'amount', 'vol']:
            if i not in market.columns:
                raise ValueError("value must include %s"%i)
        if ('pre_close' not in market.columns) and ('exfactor' not in market.columns):
            raise ValueError('value must include pre_close or exfactor')
        self.market = market
    # 计算/引用因子
    def cal_factor(self, code):
        if code in self.market.columns:
            return self.market[code]
        else:
            exp = code.split('-')
            if len(exp)==1:  # 基础因子
                self.cal_basic_factor(code)
            else:  # 时序计算（MA/EMA/Zscore/Max/Min/Std/Sum/rank/quantile/Skew/Kurt/argmin/argmax/prod)
                self.market[code] = FB.my_pd.cal_ts(self.ref(exp[0]), exp[1], int(exp[2])) 
                return self.market[code]
    # 计算/引用基础因子
    def cal_basic_factor(self, code):
        if code in self.market.columns:
             return         # 如果因子存在则不计算
        code_split = code.split('.')
        key = code_split[0]
        para = code_split[1:]
        if key=='exfactor':    # 复权因子
            exfactor = self.cal_factor('close').groupby('code').shift()/self.cal_factor('pre_close')
            self.market[code] = exfactor.groupby('code').cumprod() 
        elif key in ['ex_close', 'ex_open', 'ex_high', 'ex_low']:
            self.market[code] = self.cal_factor(code[3:])*self.cal_factor('exfactor')
        elif key=='vwap':     
            self.market[code] = self.cal_factor('amount')/self.cal_factor('vol')
        elif key=='Ret':      
            self.market[code] = self.cal_factor('ex_close')/self.cal_factor('ex_close').groupby('code').shift()
        elif key=='RSI':     
            up = pd.Series(np.where(self.cal_factor('ret')>0, self.cal_factor('ret'), 0), \
                           index=self.market.index)
            down = pd.Series(np.where(self.cal_factor('ret')<0, -self.cal_factor('ret'), 0), \
                             index=self.market.index)
            sumup = FB.my_pd.cal_ts(up, 'Sum', int(para[0]))
            sumdown = FB.my_pd.cal_ts(down, 'Sum', int(para[0]))
            rsi = sumup/(sumup+sumdown)
            self.market[code] = rsi
        elif key=='DrawDown':        # 唐奇安通道 上轨距离
            max_c = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            self.market[code] = self.cal_factor('ex_close')/min_c-1 
        elif key=='DrawUp':
            min_c = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = 1-self.cal_factor('ex_close')/max_c
        elif key=='AMP':   # 振幅
            exhigh_Max = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            exlow_Min = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = (exhigh_Max - exlow_Min)/exlow_Min  
        elif key=='TR':    # 真实波动
            self.market[code] = pd.concat([self.cal_factor('ex_high')-self.cal_factor('ex_low'),\
                abs(self.cal_factor('ex_close').groupby('code').shift()-self.cal_factor('ex_high')), \
                abs(self.cal_factor('ex_close').groupby('code').shift()-self.cal_factor('ex_low'))], axis=1).max(axis=1) 
        elif key=='Amihud':  # 流动性
            self.market[code] = self.cal_factor('Ret')/self.cal_factor('amount')
        elif key=='MACD':   #  MACD.d.d.d 
            EMAslow = FB.my_pd.cal_ts(self.cal_factor('ex_close'), 'EMA', int(para[1])) 
            EMAfast = FB.my_pd.cal_ts(self.cal_factor('ex_close'), 'EMA', int(para[0]))
            DIF = EMAfast-EMAslow
            DEM = FB.my_pd.cal_ts(DIF, 'EMA', int(para[2]))
            self.market[code] = DIF-DEM
        elif key=='RSRS':        # high/low 回归斜率  RSRS.d
            deltax = self.cal_factor('ex_low')-FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'MA', int(para[0]))
            deltay = self.cal_factor('ex_high')-FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'MA', int(para[0]))
            self.market[code] = FB.my_pd.cal_ts(deltax*deltay, 'Sum', int(para[0]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])) 
        elif key=='VolPriceCorr':  # 量价相关性   VolPriceCorr.d
            deltax = self.cal_factor('vol')-FB.my_pd.cal_ts(self.cal_factor('vol'), 'MA', int(para[0]))
            deltay = self.cal_factor('ex_close')-FB.my_pd.cal_ts(self.cal_factor('ex_close'), 'MA', int(para[0]))
            self.market[code] = (FB.my_pd.cal_ts((deltax*deltay), 'Sum', int(para[0]))/\
                    (np.sqrt(FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])))*\
                     np.sqrt(FB.my_pd.cal_ts(deltay**2, 'Sum', int(para[0]))))).fillna(0)
        return self.market[code]
