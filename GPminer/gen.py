import numpy as np
import pandas as pd
from random import shuffle,choice,sample
from GPminer import * 
from itertools import combinations


# 种群繁殖类

class Gen():
    # 种群，因子库
    def __init__(self, basket, popu0=None):
        self.basket = basket
        if popu0==None:
            self.popu = popu.Population()
        else:
            self.popu = popu0
    # 增强或减小某因子权重
    def mutation_score_dw(self):
        score0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = score0.exp
        if len(exp)==1:
            self.popu.add(score0.code)
            return {score0.code}
        random_select = np.random.randint(len(exp))
        # 增加或减少权重，避免因子权重减为0
        if (np.random.rand()>0.5)|(exp[random_select][2]==1):
            exp[random_select][2] = exp[random_select][2]+1
        else:
            exp[random_select][2] = exp[random_select][2]-1
        score_new = ind.Score(exp)
        self.popu.add(score_new.code)
        return {score_new.code}
    # 增减因子,增加因子时随机给一个已存在因子的权重
    def mutation_score_and(self):
        score0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = score0.exp
        random_select0 = np.random.randint(len(exp))
        if (np.random.rand()>0.5)&(len(exp)!=1):
            exp.pop(random_select0)
        else:
            random_select = np.random.randint(len(self.basket))
            # 随机赋一个已有因子的权重
            exp.append([self.basket[random_select], np.random.rand()>0.5, \
                         exp[random_select0][2]])
        score_new = ind.Score(exp)
        self.popu.add(score_new.code)
        return {score_new.code}
    # 替换因子
    def mutation_score_replace(self):
        score0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = score0.exp
        random_select0 = np.random.randint(len(exp))
        random_select = np.random.randint(len(self.basket))
        already = [i[0] for i in exp]
        # 不能替换前后相同，不能替换已有因子
        while (exp[random_select0][0]==self.basket[random_select])|\
                        (self.basket[random_select] in already):
            random_select = np.random.randint(len(self.basket))
        exp[random_select0][0] = self.basket[random_select]
        score_new = ind.Score(exp)
        self.popu.add(score_new.code)
        return {score_new.code}
    # 两因子交叉
    def cross_score_exchange(self):
        if len(self.popu.codes)<2:
            #print('种群规模过小')
            return 0
        sele = list(self.popu.subset(2).codes)
        exp0 = self.popu.type(sele[0]).exp
        exp1 = self.popu.type(sele[1]).exp
        # 需要打乱顺序，保证交叉的多样性
        shuffle(exp0)
        shuffle(exp1)
        # 如果有一个是单因子的话则合成一个因子
        if (len(exp0)==1)|(len(exp1)==1):
            score_new = ind.Score(exp0+exp1)
            self.popu.add(score_new.code)
            return {score_new.code}
        cut0 = np.random.randint(1,len(exp0))
        cut1 = np.random.randint(1,len(exp1))
        score_new_0 = ind.Score(exp0[:cut0]+exp1[:cut1])
        score_new_1 = ind.Score(exp0[cut0:]+exp1[cut1:])
        self.popu.add({score_new_0.code, score_new_1.code})
        return {score_new_0.code, score_new_1.code}
    # 遍历单因子、双因子、三因子, 作为种子组合
    def get_seeds(self):
        if self.popu.type==(ind.Score):
            popu0 = popu.Population() 
            for i in self.basket:
                popu0.add({ind.Score([[i, True, 1]]).code, ind.Score([[i, False, 1]]).code})
            for i,j in list(combinations(self.basket, 2)):
                for b0 in [True, False]:
                    for b1 in [True, False]:
                        popu0.add(ind.Score([[i, b0, 1], [j, b1, 1]]).code)
            for i,j,k in list(combinations(self.basket, 3)):
                for b0 in [True, False]:
                    for b1 in [True, False]:
                        for b2 in [True, False]:
                            popu0.add(ind.Score([[i, b0, 1], [j, b1, 1], [k, b2, 1]]).code)
        return popu0.codes 
    # 种群繁殖
    def multiply(self, multi=2, prob_dict={}):
        # 各算子被执行的概率，如果空则全部算子等概率执形
        if prob_dict=={}:
            opts = [f for f in dir(Gen) if 'mutation' in f or 'cross' in f]
            prob_ser = pd.Series(np.ones(len(opts)), index=opts)
        else:
            prob_ser = pd.Series(prob_dict.values(), index=prob_dict.keys())
        prob_ser = prob_ser/prob_ser.sum()
        prob_ser = prob_ser.cumsum()
        popu_size = len(self.popu.codes)
        while len(self.popu.codes)<int(popu_size*multi):
            r = np.random.rand()
            for func,v in prob_ser.items():
                if r<v:
                    break
                getattr(self, func)()
