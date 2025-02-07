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
        self.market = market
        for i in ['close', 'open', 'high', 'low', 'pre_close', 'amount', 'vol']:
            if i not in market.columns:
                raise ValueError("value must include %s"%i)
        if 'exfactor' not in market.columns:
            self.cal_factor('exfactor')
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
        # 量价基础 basic
        elif key in ['ex_close', 'ex_open', 'ex_high', 'ex_low']:
            self.market[code] = self.cal_factor(code[3:])*self.cal_factor('exfactor')
        elif key=='vwap':     
            self.market[code] = self.cal_factor('amount')/self.cal_factor('vol')
        # 动量反转  mom
        elif key=='Ret':      
            self.market[code] = self.cal_factor('close')/self.cal_factor('pre_close')
        elif key=='nightRet':     # 隔夜收益（开盘跳空
            self.market[code] = self.cal_factor('open')/self.cal_factor('pre_close')-1
        elif key=='T0Ret':
            self.market['T0Ret'] = self.cal_factor('close')/self.cal_factor('open')-1
        elif key=='maxRet':
            self.market['maxRet'] = self.cal_factor('high')/self.cal_factor('low')-1
        elif key=='bodyRatio':   # k线实体占比
            self.market[code] = abs(self.cal_factor('close')-self.cal_factor('open'))/(self.cal_factor('high')-self.cal_factor('low'))
        elif key=='RSI':     
            up = pd.Series(np.where(self.cal_factor('Ret')>0, self.cal_factor('Ret'), 0), \
                           index=self.market.index)
            down = pd.Series(np.where(self.cal_factor('Ret')<0, -self.cal_factor('Ret'), 0), \
                             index=self.market.index)
            sumup = FB.my_pd.cal_ts(up, 'Sum', int(para[0]))
            sumdown = FB.my_pd.cal_ts(down, 'Sum', int(para[0]))
            rsi = sumup/(sumup+sumdown)
            self.market[code] = rsi
        elif key=='DrawDown':        # 唐奇安通道 上轨距离   DrawDown.d
            max_c = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            self.market[code] = self.cal_factor('ex_close')/min_c-1 
        elif key=='DrawUp':   # 唐奇安通道 下轨距离  DrawUp.d
            min_c = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = 1-self.cal_factor('ex_close')/max_c
        # 波动率  volatility
        elif key=='AMP':   # 振幅   AMP.d
            exhigh_Max = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            exlow_Min = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = (exhigh_Max - exlow_Min)/exlow_Min
        elif key=='TR':    # 真实波动
            self.market[code] = pd.concat([self.cal_factor('high')-self.cal_factor('low'),\
                abs(self.cal_factor('pre_close')-self.cal_factor('high')), \
                abs(self.cal_factor('pre_close')-self.cal_factor('low'))], axis=1).max(axis=1) 
        # 成交流动性   vol
        elif key=='Amihud':  # 流动性
            self.market[code] = self.cal_factor('Ret')/self.cal_factor('amount')
        elif key=='UnusualVol':    # 异常成交量  UnusualVol.d
            self.market[code] = self.cal_factor('vol')/self.cal_factor('vol_MA_'+para[0])
        # 技术形态  kbars
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
        # 量价相关  composite
        elif key in ['VolVWAPCorr', 'VolAMPCorr', 'VolRetCorr']:  # 量价相关性   VolPriceCorr.d
            deltax = self.cal_factor('vol')-FB.my_pd.cal_ts(self.cal_factor('vol'), 'MA', int(para[0]))
            if key=='VolVWAPCorr':            
                deltay = self.cal_factor('vwap')-FB.my_pd.cal_ts(self.cal_factor('vwap'), 'MA', int(para[0]))
            elif key=='VolAMPCorr':
                deltay = self.cal_factor('AMP.1')-FB.my_pd.cal_ts(self.cal_factor('AMP.1'), 'MA', int(para[0]))
            elif key=='VolRetCorr':
                deltay = self.cal_factor('Ret')-FB.my_pd.cal_ts(self.cal_factor('Ret'), 'MA', int(para[0]))
            self.market[code] = (FB.my_pd.cal_ts((deltax*deltay), 'Sum', int(para[0]))/\
                    (np.sqrt(FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])))*\
                     np.sqrt(FB.my_pd.cal_ts(deltay**2, 'Sum', int(para[0]))))).fillna(0)
        # 金融学理论   model        
        elif key=='beta':    # CAMP理论 beta/alpha/estd  beta.d
            # 计算市场平均收益
            deltax = self.cal_factor('meanRet')-FB.my_pd.cal_ts(self.cal_factor('meanRet'), 'MA', int(para[0]))
            deltay = self.cal_factor('Ret')-FB.my_pd.cal_ts(self.cal_factor('Ret'), 'MA', int(para[0]))
            self.market[code] = FB.my_pd.cal_ts(deltax*deltay, 'Sum', int(para[0]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])) 
        elif key=='oc2hl':   # 开收重心到低高重心变动
            self.market[code] = (self.cal_factor('high')+self.cal_factor('low'))/(self.cal_factor('open')+self.cal_factor('close')) 
        return self.market[code]
