import pandas as pd
from random import choice,sample
import re, ind



# 种群

class Population():
    # 个体类型
    def __init__(self, type=ind.Score):
        self.type = type   # ind.Score, ind.Pool
        # 使用一个集合来储存code值
        self.codes = set()
    def code2exp(self, code):
        return self.type(code)
    def add(self, code):
        if type(code)!=type(set()):
            self.codes = self.codes|{code}
        else:
            self.codes = self.codes|code
    def sub(self, code):
        if type(code)!=type(set()):
            self.codes = self.codes-{code}
        else:
            self.codes = self.codes-code
    def reset(self, codes):
        self.codes = set()
        for code in codes:
            self.codes = self.codes|{self.type(code).code}
    def get_name(self, n=3):
        factor_count = {}   # 因子出现频率
        for i in self.codes:
            if self.type==ind.Score:
                for j in i.split('+'):
                    split = j.split('*')
                    name = '·'.join(split[1:])
                    if name not in factor_count.keys():
                        factor_count[name] = int(split[0])
                    else:
                        factor_count[name] = factor_count[name]+int(split[0])
            elif self.type==ind.Pool:
                for j in i.split('|'):
                    name = re.findall("^(.*?)[><=]", j)[0]
                    if name not in factor_count.keys():
                        factor_count[name] = 1 
                    else:
                        factor_count[name] = factor_count[name]+1
        factor_count = pd.Series(factor_count.values(), index=factor_count.keys())\
                .sort_values(ascending=False)
        self.name = ';'.join(factor_count.index[:n])
    # 从群体中采样
    def subset(self, size=1):
        popu0 = Population(self.type)
        popu0.add(set(sample(list(self.codes), size)))
        return popu0
