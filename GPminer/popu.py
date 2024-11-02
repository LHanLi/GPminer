import pandas as pd
from random import choice,sample
import GPminer as GPm
import re


# 种群
class Population():
    # 使用一个集合来储存code值
    def __init__(self, type=GPm.ind.Score):
        self.type = type   
        self.codes = set()
    def code2exp(self, code):
        return self.type(code)
    # 默认不检查code（背后代表的个体）是否重复
    def add(self, code, check=False):
        if type(code)!=type(set()):
            if check:
                self.codes = self.codes|{self.type(code).code}
            else:
                self.codes = self.codes|{code}
        else:
            if check:
                for c in code:
                    self.codes = self.codes|{self.type(c).code}
            else:
                self.codes = self.codes|code
    def sub(self, code, check=False):
        if type(code)!=type(set()):
            self.codes = self.codes-{self.type(code).code}
        else:
            if check:
                for c in code:
                    self.codes = self.codes|{self.type(c).code}
            else:
                self.codes = self.codes|code
    def reset(self, code, check=False):
        self.codes = set()
        self.add(code, check)
    def get_name(self, n=3):
        factor_count = {}   # 因子出现频率
        for i in self.codes:
            if self.type==GPm.ind.Score:
                for j in i.split('+'):
                    split = j.split('*')
                    name = '·'.join(split[1:])
                    if name not in factor_count.keys():
                        factor_count[name] = int(split[0])
                    else:
                        factor_count[name] = factor_count[name]+int(split[0])
            elif self.type==GPm.ind.Pool:
                for j in i.split('|'):
                    name = re.findall("^(.*?)[><=]", j)[0]
                    if name not in factor_count.keys():
                        factor_count[name] = 1 
                    else:
                        factor_count[name] = factor_count[name]+1
        factor_count = pd.Series(factor_count.values(), index=factor_count.keys())\
                .sort_values(ascending=False)
        self.name = ';'.join(factor_count.index[:n])
        return self.name
    # 从群体中采样
    def subset(self, size=1):
        popu0 = Population(self.type)
        popu0.add(set(sample(list(self.codes), size)))
        return popu0
    # 从群体中采样
    def pop(self, size=1): 
        if size==1:



