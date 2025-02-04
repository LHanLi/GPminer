import GPminer as GPm
import pandas as pd


# 因子命名格式：
# factor_method_period
# factor: 基础因子名， 如果有参数，用^分割
# method：时序算符（MA/EMA/Zscore/Max/Min/Std/Sum/\
# rank/quantile/Skew/Kurt/argmin/argmax/prod), period: 时序算符周期
class Factor():
    def __init__(self, market, exfactor):
        self.market = market
        self.exfactor = exfactor
    # 计算因子
    def cal_factor(self, code):
        exp = code.split('_')
        if len(exp)==1:
            self.cal_basic_factor(code)
        else:
            pass 
    # 计算基础因子
    def cal_basic_factor(self, code):
        pass


