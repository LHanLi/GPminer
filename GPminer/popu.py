import pandas as pd
from random import choice,sample
import GPminer as GPm


# 种群
class Population():
    # 使用一个集合来储存code值, fix是一个个体，为该种群所有个体公用的个体，默认为空
    def __init__(self, type=GPm.ind.Score, fix_ind=None):
        self.type = type
        self.codes = set()
        self.fix_ind = fix_ind
    def code2exp(self, code):
        return self.type(code)
    # 附加上固定属性
    def add_fix(self, code):
        if self.fix_ind==None:
            return code
        elif '&' in code:  # SP
            s, p = code.split('&')
            s_, p_ = self.fix_ind.code.split('&')
            return self.add_fix(code.split('&')[0]) + '&' + self.add_fix(code.split('&')[1])
        elif ';' in code:  # Pool/Pooland
            _ = self.fix_ind.pool.code.split(';')
            __ = code.split(';')
            __[0] = ('|' if (_[0]!='')&(__[0]!='') else '') + __[0]
            __[1] = ('|' if (_[1]!='')&(__[1]!='') else '') + __[1]
            return _[0]+__[0]+';'+_[1]+__[1]
        else:        # Score
            if code=='':
                return self.fix_ind.score.code
            elif self.fix_ind.score.code=='':
                return code
            else:
                return code + '+' + self.fix_ind.score.code
    # 默认单个输入的时候检查ind是否重复，输入集合时不检查
    def add(self, code):
        if type(code)!=type(set()):
            self.codes = self.codes|{self.type(self.add_fix(code)).code}
        else:
            self.codes = self.codes|code
    def sub(self, code):
        if type(code)!=type(set()):
            self.codes = self.codes-{self.type(code).code}
        else:
            self.codes = self.codes-code
    def reset(self, code):
        self.codes = set()
        self.add(code)
    def get_name(self, n=3):
        factor_count = pd.Series()  # 因子出现频率
        for i in self.codes:
            factor_count = factor_count.add(self.type(i).factors(), fill_value=0)
        self.name = ';'.join(factor_count.sort_values(ascending=False).index[:n]) 
        return self.name
    # 从群体中采样
    def subset(self, size=1):
        if size==1:
            return self.type(sample(list(self.codes), 1)[0])
        popu0 = Population(self.type)
        popu0.add(set(sample(list(self.codes), size)))
        return popu0


