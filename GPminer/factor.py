import GPminer as GPm
import pandas as pd
import numpy as np
import FreeBack as FB


# 因子命名格式：
# factor-method-period
# factor: 基础因子名， 如果有参数，用.分割
# method：时序算符, period: 时序算符周期
class Factor():
    # type: stock/bond/future/crypto
    def __init__(self, market, type='stock'):
        self.market = market
        self.type = type
        for i in ['close', 'open', 'high', 'low', 'pre_close', 'amount', 'vol']:
            if i not in market.columns:
                raise ValueError("value must include %s"%i)
        if 'exfactor' not in market.columns:
            self.cal_factor('exfactor')
    # 计算/引用因子
    def cal_factor(self, code, ifrecal=False):
        if (code in self.market.columns)&(not ifrecal):
            return self.market[code]      # 如果因子存在则不计算
        else:
            # 因子分解为基础因子名和时序运算符
            exp = code.split('-')
            if len(exp)==1:  # 基础因子
                self.cal_basic_factor(code)
            else:  # 时序计算（MA/EMA/Zscore/Max/Min/Std/Sum/rank/quantile/Skew/Kurt/argmin/argmax/prod)
                self.market[code] = FB.my_pd.cal_ts(self.cal_factor(exp[0]), exp[1], int(exp[2])) 
            # 返回计算因子
            if code in self.market.columns:
                return self.market[code] 
            else:
                print('failed cal')
    # 计算/引用基础因子
    def cal_basic_factor(self, code):
        # 因子分解为因子名和参数
        code_split = code.split('.')
        key = code_split[0]
        para = code_split[1:]
        ##############################################################################
        ########################### 交易规则类 rules ##################################
        ##############################################################################
        if key=='PriceLimit':
        # 涨跌停
            def get_PriceLimit():
                    # 使用长度为3的字符串表示涨跌停状态 
                    # 开盘收盘：+表示涨停 -表示跌停 0表示正常
                    # 盘中: 0盘中时表示非一直跌停或涨停，+或-表示存在涨停或跌停
                    date = self.market.index.get_level_values(0)
                    global get_limit
                    if self.type=='stock':
                        mindelta = 0.01
                        def get_limit(up=True):  # 获取涨跌幅度限制    简化处理，详细规则见:xueqiu.com/4610950050/276325750
                            return np.select([self.cal_factor('place_sector')=='上交所.科创板', self.cal_factor('place_sector')=='深交所.创业板', self.cal_factor('place_sector')=='北交所.北证'],\
                                    [np.where(self.cal_factor('dur_tradedays')<=5, 999 if up else 1, 0.2),\
                                    np.where(date>=pd.to_datetime('2020-8-24'), np.where(self.cal_factor('dur_tradedays')<=5, 999 if up else 1, 0.2),\
                                        np.where(self.cal_factor('special_type').isin(['*ST', 'ST', 'delist']), 0.05, \
                                            np.where(date>=pd.to_datetime('2014-1-1'), np.where(self.cal_factor('dur_tradedays')==1, 0.44 if up else 0.36, 0.1),\
                                                np.where(self.cal_factor('dur_tradedays')==1, 999 if up else 1, 0.1)))),\
                                    np.where(self.cal_factor('dur_tradedays')==1, 999 if up else 1, 0.3)], \
                                    np.where(self.cal_factor('special_type').isin(['*ST', 'ST', 'delist']), 0.05, \
                                        np.where(date>=pd.to_datetime('2023-4-10'), np.where(self.cal_factor('dur_tradedays')<=5, 999 if up else 1, 0.1),\
                                        np.where(date>=pd.to_datetime('2014-1-1'), np.where(self.cal_factor('dur_tradedays')<=5, 0.44 if up else 0.36, 0.1),\
                                            np.where(self.cal_factor('dur_tradedays')==1, 999 if up else 1, 0.1)))))
                    elif self.type=='bond':
                        mindelta = 0.001
                        def get_limit(up=True):  # 获取涨跌幅度限制    简化处理，详细规则见:xueqiu.com/4610950050/276325750
                            return np.where(date<=pd.to_datetime('2022-8-1'), 999 if up else 1,\
                                    np.where(self.cal_factor('dur_tradedays')==1, 0.3 if up else 0.433, 0.2))
                    uplimit = get_limit()
                    downlimit = get_limit(False)
                    ifonceuplimit = self.cal_factor('high')+mindelta>self.cal_factor('pre_close')*(1+uplimit)
                    ifoncedownlimit = self.cal_factor('low')-mindelta<self.cal_factor('pre_close')*(1-downlimit)
                    ifcloseuplimit = self.cal_factor('close')+mindelta>self.cal_factor('pre_close')*(1+uplimit)
                    ifclosedownlimit = self.cal_factor('close')-mindelta<self.cal_factor('pre_close')*(1-downlimit)
                    ifopenuplimit = self.cal_factor('open')+mindelta>self.cal_factor('pre_close')*(1+uplimit)
                    ifopendownlimit = self.cal_factor('open')-mindelta<self.cal_factor('pre_close')*(1-downlimit)
                    return pd.Series(np.where(ifopenuplimit, '+', np.where(ifopendownlimit, '-', 0)), index=self.market.index) +\
                        pd.Series(np.where((~ifopendownlimit)&(~ifclosedownlimit)&ifoncedownlimit, '-', \
                            np.where((~ifopenuplimit)&(~ifcloseuplimit)&ifonceuplimit, '+', 0)), index=self.market.index) +\
                            pd.Series(np.where(ifcloseuplimit, '+', np.where(ifclosedownlimit, '-', 0)), index=self.market.index) # 开盘+盘中+收盘
            self.market[code] = get_PriceLimit() 
        ##############################################################################
        ########################### 公告 announcements ##################################
        ##############################################################################
        elif key=='special_type':
            def get_special_type(name):
                if '*ST' in name:
                    return '*ST'
                elif 'ST' in name:
                    return 'ST'
                elif '退' in name:
                    return 'delist'
                else:
                    return 'Normal'
            self.market[code] = self.cal_factor('name').map(lambda x: get_special_type(x))
        elif key=='times':   # times.a.d 最近d日公告a的次数 d
            self.market[code] = self.cal_factor(para[0])
        elif key=='days':   # days.a 距离上次为True天数
            dayscode = self.market[self.cal_factor(para[0])][[]].reset_index()
            dayscode['anndate'] = dayscode['date']
            alldayscode = self.market[[]].reset_index()
            days = alldayscode.merge(dayscode, on=['date', 'code'], how='left')
            days = days.set_index(['date', 'code']).groupby('code')['anndate'].ffill()
            days = days.reset_index()
            days[code] = ((days['date']-days['anndate']).dt.days+1).fillna(999)
            self.market[code] = days.set_index(['date', 'code'])[code]
        elif key=='tradedays':
            alltradeday = self.market.index.get_level_values(0).unique()
            def count_tradedays(start, end):
                if (start not in alltradeday)|(end not in alltradeday):
                    return np.nan
                start_index = np.searchsorted(alltradeday, start, side='left')
                end_index = np.searchsorted(alltradeday, end, side='right')
                return end_index-start_index  # 第一天为1
            dayscode = self.market[self.cal_factor(para[0])][[]].reset_index()
            dayscode['anndate'] = dayscode['date']
            alldayscode = self.market[[]].reset_index()
            days = alldayscode.merge(dayscode, on=['date', 'code'], how='left')
            days = days.set_index(['date', 'code']).groupby('code')['anndate'].ffill()
            days = days.reset_index()
            days[code] = np.vectorize(count_tradedays)(days['anndate'], days['date'])
            self.market[code] = days.set_index(['date', 'code'])[code].fillna(999)
        ##############################################################################
        ########################### 财务数据 finance ##################################
        ##############################################################################
        elif key=='现金':
            self.market['现金'] = self.cal_factor('currency')/1e8
        elif key=='总资产':
            self.market['总资产'] = self.cal_factor('asset')/1e8
        elif key=='BP':
            self.market['BP'] = self.cal_factor('equity')/(self.cal_factor('close')*\
                self.cal_factor('total_shares'))  # 市净率倒数
        elif key=='EP':
            self.market['EP'] = self.cal_factor('profit')/(self.cal_factor('close')*self.cal_factor('total_shares'))  # 市盈率倒数 ttm
        elif key=='SP':
            self.market['SP'] = self.cal_factor('revenue')/(self.cal_factor('close')*self.cal_factor('total_shares'))  # 市销率倒数 ttm
        elif key=='ROE':
            self.market['ROE'] = self.cal_factor('profit-ttm')/self.cal_factor('equity')
        elif key=='ROA':
            self.market['ROA'] = self.cal_factor('profit-ttm')/self.cal_factor('asset')
        elif key=='资产负债率':
            self.market['资产负债率'] = self.cal_factor('borrow')/self.cal_factor('asset')  # 资产负债率
        elif key=='有形资产负债率':
            self.market['有形资产负债率'] = self.cal_factor('borrow')/\
                (self.cal_factor('asset')-self.cal_factor('intangible'))  # 资产负债率
        elif key=='股息率':
            self.market['股息率'] = self.cal_factor('dividends_ttm')/(self.cal_factor('close')*self.cal_factor('total_shares')) # 股息率
        elif key=='流动比率':
            self.market['流动比率'] = self.cal_factor('current_asset')/self.cal_factor('current_borrow')
        elif key=='速动比率':
            self.market['速动比率'] = (self.cal_factor('current_asset')-self.cal_factor('inventory'))/self.cal_factor('current_borrow')
        elif key=='存货周转天数':
            self.market['存货周转天数'] = 360*self.cal_factor('inventory-ttm')/self.cal_factor('revenue-ttm')
        elif key=='应收账款周转天数':
            self.market['应收账款周转天数'] = 360*self.cal_factor('receivable-ttm')/self.cal_factor('revenue-ttm')
        elif key=='营业周期':
            self.market['营业周期'] = self.cal_factor('存货周转天数') + self.cal_factor('应收账款周转天数')
        elif key=='流动资产周转率':
            self.market['流动资产周转率'] = self.cal_factor('revenue-ttm')/self.cal_factor('current_asset-ttm')
        elif key=='总资产周转率':
            self.market['总资产周转率'] = self.cal_factor('revenue-ttm')/self.cal_factor('asset-ttm')
        elif key=='已获利息倍数':
            self.market['已获利息倍数'] = self.cal_factor('profit-ttm')/self.cal_factor('dividends_ttm')
        elif key=='净利率':
            self.market['净利率'] = self.cal_factor('profit-ttm')/self.cal_factor('revenue-ttm')
        elif key=='毛利率':
            self.market['毛利率'] = (self.cal_factor('revenue-ttm') - self.cal_factor('operating_cost-ttm'))/self.cal_factor('revenue-ttm')
        elif key=='现金到期债务比':
            self.market['现金到期债务比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/\
                (self.cal_factor('st_borrow')+self.cal_factor('1y_borrow'))
        elif key=='现金流动负债比':
            self.market['现金流动负债比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('current_borrow'))
        elif key=='现金负债比':
            self.market['现金负债比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('borrow'))
        elif key=='现金销售比':
            self.market['现金销售比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('revenue'))
        elif key=='全部资产现金回收率':
            self.market['全部资产现金回收率'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('asset'))
        elif key=='营收同比':
            self.market['营收同比'] = self.cal_factor('revenue-YOY')
        elif key=='净利润同比':
            self.market['净利润同比'] = self.cal_factor('profit-YOY')
        ##############################################################################
        ################################ 市值 Cap ##################################
        ##############################################################################
        elif key=='Cap':
             self.market[code] = self.cal_factor('total_shares')*self.cal_factor('close')/1e8
        elif key=='floatCap':
             self.market[code] = self.cal_factor('float_shares')*self.cal_factor('close')/1e8
        elif key=='freeCap':
             self.market[code] = self.cal_factor('free_float_shares')*self.cal_factor('close')/1e8
        ##############################################################################
        ################################ 价格因子 price ##############################
        ##############################################################################
        elif key=='vwap': 
            self.market[code] = (self.cal_factor('amount')/self.cal_factor('vol')).\
                fillna(self.market['close'])
        elif key=='exfactor':    # 复权因子
            exfactor = self.cal_factor('close').groupby('code').shift()/\
                self.cal_factor('pre_close')
            self.market[code] = exfactor.groupby('code').cumprod()
        elif key in ['ex_close', 'ex_open', 'ex_high', 'ex_low']:
            self.market[code] = self.cal_factor(code[3:])*self.cal_factor('exfactor')
        ##############################################################################
        ################################ k线形态 kbar ################################
        ##############################################################################
        elif key=='Ret':      
            self.market[code] = self.cal_factor('close')/self.cal_factor('pre_close')-1
        elif key=='highRet':     
            self.market[code] = self.cal_factor('high')/self.cal_factor('pre_close')-1
        elif key=='lowRet':     
            self.market[code] = self.cal_factor('low')/self.cal_factor('pre_close')-1
        elif key=='nightRet':     # 隔夜收益（开盘跳空
            self.market[code] = self.cal_factor('open')/self.cal_factor('pre_close')-1
        elif key=='T0Ret':
            self.market['T0Ret'] = self.cal_factor('close')/self.cal_factor('open')-1
        elif key=='maxRet':
            self.market['maxRet'] = self.cal_factor('high')/self.cal_factor('low')-1
        elif key=='bodyRatio':   # k线实体占比
            self.market[code] = abs(self.cal_factor('close')-self.cal_factor('open'))/\
                (self.cal_factor('high')-self.cal_factor('low'))
        elif key=='oc2hl':   # 开收重心到低高重心变动
            self.market[code] = (self.cal_factor('high')+self.cal_factor('low'))/\
                (self.cal_factor('open')+self.cal_factor('close')) 
        elif key=='MACD':   #  MACD.d.d.d 
            EMAslow = FB.my_pd.cal_ts(self.cal_factor('ex_close'), 'EMA', int(para[1])) 
            EMAfast = FB.my_pd.cal_ts(self.cal_factor('ex_close'), 'EMA', int(para[0]))
            DIF = EMAfast-EMAslow
            DEM = FB.my_pd.cal_ts(DIF, 'EMA', int(para[2]))
            self.market[code] = DIF-DEM
        elif key=='RSI':    # RSI.d  d日RSI 
            up = pd.Series(np.where(self.cal_factor('Ret')>0, self.cal_factor('Ret'), 0), \
                           index=self.market.index)
            down = pd.Series(np.where(self.cal_factor('Ret')<0, -self.cal_factor('Ret'), 0), \
                             index=self.market.index)
            sumup = FB.my_pd.cal_ts(up, 'Sum', int(para[0]))
            sumdown = FB.my_pd.cal_ts(down, 'Sum', int(para[0]))
            rsi = sumup/(sumup+sumdown)
            self.market[code] = rsi
        elif key=='RSRS':        # high/low 回归斜率  RSRS.d
            deltax = self.cal_factor('ex_low')-FB.my_pd.cal_ts(self.cal_factor('ex_low'), \
                                                                'MA', int(para[0]))
            deltay = self.cal_factor('ex_high')-FB.my_pd.cal_ts(self.cal_factor('ex_high'), \
                                                            'MA', int(para[0]))
            self.market[code] = FB.my_pd.cal_ts(deltax*deltay, 'Sum', \
                            int(para[0]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])) 
        ##############################################################################
        ########h##################### 动量反转 mom ##################################
        ##############################################################################
        elif key=='deviation':         # deviation.d  d日均线乖离率
            self.market[code] = self.cal_factor('ex_close')/\
                self.cal_factor('ex_close-MA-%s'%para[0])-1
        elif key=='DrawDown':        # 唐奇安通道 上轨距离   DrawDown.d
            max_c = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            self.market[code] = 1-self.cal_factor('ex_close')/max_c
        elif key=='DrawUp':   # 唐奇安通道 下轨距离  DrawUp.d
            min_c = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = self.cal_factor('ex_close')/min_c-1 
        ##############################################################################
        ########################### 波动率 volatility ################################
        ##############################################################################
        elif key=='AMP':   # 振幅   AMP.d
            exhigh_Max = FB.my_pd.cal_ts(self.cal_factor('ex_high'), 'Max', int(para[0]))
            exlow_Min = FB.my_pd.cal_ts(self.cal_factor('ex_low'), 'Min', int(para[0]))
            self.market[code] = (exhigh_Max - exlow_Min)/exlow_Min
        elif key=='TR':    # 真实波动
            self.market[code] = pd.concat([self.cal_factor('high')-self.cal_factor('low'),\
                abs(self.cal_factor('pre_close')-self.cal_factor('high')), \
                abs(self.cal_factor('pre_close')-self.cal_factor('low'))], axis=1).max(axis=1) 
        elif key=='UpTimes':      # UpTimes.a.d 过去d日涨幅超过a%天数
            self.market[code] = FB.my_pd.cal_ts((self.cal_factor('highRet')>float(para[0])/100),\
                                        'Sum', int(para[1]))
        elif key=='DownTimes':      # *.a.d 过去d日最大跌幅超过a%天数
            self.market[code] = FB.my_pd.cal_ts((self.cal_factor('lowRet')<float(para[0])/100),\
                                        'Sum', int(para[1]))
        elif key=='ampTimes':      # *.a.d 过去d日最大波动超过a%天数
            self.market[code] = FB.my_pd.cal_ts(self.cal_factor('maxRet')>float(para[0]),\
                                        'Sum', int(para[1]))
        elif key=='LimitTimes':      # LimitTimes.d d日涨跌停天数
            self.market[code] = FB.my_pd.cal_ts(self.cal_factor('PriceLimit').\
                    map(lambda x: ('+' in x)|('-' in x)), 'Sum', int(para[0]))
        ##############################################################################
        ######################## 交易活跃度、流动性 volhot #############################
        ##############################################################################
        elif key=='turnover':  
            self.market[code] = self.cal_factor('vol')/self.cal_factor('free_float_shares')
        elif key=='Amihud':  # 流动性
            self.market[code] = self.cal_factor('Ret')/self.cal_factor('amount')
        elif key=='UnusualVol':    # 异常成交量  UnusualVol.d
            self.market[code] = self.cal_factor('vol')/self.cal_factor('vol-MA-'+para[0])-1
        ##############################################################################
        ################################ 量价相关 corr ################################
        ##############################################################################
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
        ##############################################################################
        ############################# 经济学理论 model ################################
        ##############################################################################
        elif key=='beta':    # CAMP理论 beta/alpha/estd  beta.d
            # 计算市场平均收益
            deltax = self.cal_factor('meanRet')-FB.my_pd.cal_ts(self.cal_factor('meanRet'), 'MA', int(para[0]))
            deltay = self.cal_factor('Ret')-FB.my_pd.cal_ts(self.cal_factor('Ret'), 'MA', int(para[0]))
            self.market[code] = FB.my_pd.cal_ts(deltax*deltay, 'Sum', int(para[0]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[0])) 
        ##############################################################################
        ############################# 创新因子 fancy ################################
        ##############################################################################
        else:
            pass
        if code not in self.market.columns:
            print('warning! not basic factor %s'%code)