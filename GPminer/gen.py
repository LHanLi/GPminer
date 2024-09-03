import numpy as np
import pandas as pd
from random import shuffle,choice,sample
from GPminer import * 
from itertools import combinations
import time

# 种群繁殖类

class Gen():
    # 种群，因子库
    def __init__(self, basket=[], popu0=None, market=None):
        self.basket = basket
        if popu0==None:
            self.popu = popu.Population()
        else:
            self.popu = popu0
        # 初始化pool的参数域，需要输入market
        if self.popu.type==ind.Pool:
            self.para_space = {}
            for factor in basket:
                try:
                    self.para_space[factor] = (False, [market[factor].quantile(i) \
                                                       for i in np.linspace(0.01,0.99,21)])
                except:
                    self.para_space[factor] = (True,list(market[factor].unique())) 
    # 增强或减小某因子参数
    def mutation_score_d(self):
        score0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = score0.exp
        # 单因子权重无法改变
        if len(exp)==1:
            self.popu.add(score0.code)
            return {score0.code}
        random_select = np.random.randint(len(exp)) 
        #print('选择变异第%s个因子权重'%random_select)
        deltawmax = 0.05 # 权重改变幅度小于此阈值
        deltawmin = 0.02 # 权重改变幅度大于此阈值
        max_step = 100 # 最大的新权重组合寻找步数
        sumw = sum([i[2] for i in exp])
        wbefore = exp[random_select][2]/sumw
        mul = 1  # 全部权重乘mul，random_select权重+1, d>0增加权重，d<0减小权重 
        # 减小/增大乘数(d为负时相反）， 是/否操作exp 
        def get_wafter(method=True, opt=False):
            if method:
                if opt:
                    for i in range(len(exp)):
                        exp[i][2] = exp[i][2]*mul-d
                    exp[random_select][2] = exp[random_select][2]+d
                return (mul*exp[random_select][2])/(mul*sumw-(len(exp)-1)*d)
            else:
                if opt:
                    for i in range(len(exp)):
                        exp[i][2] = exp[i][2]*mul
                    exp[random_select][2] = exp[random_select][2]+d
                return (mul*exp[random_select][2]+d)/(mul*sumw+d)
        minw = min([i[2] for i in exp])
        if np.random.rand()>0.5:
            d = 1
            # 如果最小数字*mul大于d则通过减小乘数增大权重，否则选择通过增大系数增大权重
            wafter = get_wafter(minw*mul>d, False)
            step = 0
            while ((wafter-wbefore<deltawmin)|(wafter-wbefore>deltawmax))&(step<max_step): 
                if (wafter-wbefore)<deltawmax:
                    # 权重变化太小增大d
                    d+=1
                else:
                    # 权重变化太大增大mul
                    mul+=1
                step += 1
                # print(mul, d)
                wafter = get_wafter(minw*mul>d, False)
            get_wafter(minw*mul>d, True)
        else:
            d = -1
            # 如果最小数字*mul小于等于-d的话选择通过增大系数减小权重(d<0)
            wafter = get_wafter(minw*mul<=-d, False)
            step = 0
            while ((wbefore-wafter<deltawmin)|(wbefore-wafter>deltawmax))&(step<max_step): 
                if (wbefore-wafter)<deltawmax:
                    # 权重变化太小减小d
                    d-=1
                else:
                    # 权重变化太大增大mul
                    mul+=1
                    step += 1
                wafter = get_wafter(minw*mul<=-d, False)
            get_wafter(minw*mul<=-d, True)
            #print('通过%s系数, 减小权重, mul=%s, d=%s'%\
            #      ((lambda x: '减小' if x else '增大')(method), mul, d))
        score_new = ind.Score(exp)
        self.popu.add(score_new.code)
        return {score_new.code}
    def mutation_pool_d(self):
        pool0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = pool0.exp
        # 等概率选择一个因子（只改变一个点位）进行改变
        random_factor = choice(list(set([i[1] for i in exp])))
        random_loc = choice([i for i in range(len(exp)) if exp[i][1]==random_factor])
        print(random_factor, random_loc)
        # 如果是离散变量随机去掉或者增加值
        if self.para_space[random_factor][0]:
            if np.random.rand()>0.5:
                exp.append(['equal', random_factor, choice(self.para_space[random_factor][1])])
            else:
                exp.pop(random_loc)
        else:
            # 小于等于最小值或大于等于最大值则变异为次小和次大的
            # 其他情况时50%几率变大，50%几率变小 
            if exp[random_loc][2]<=self.para_space[random_factor][1][0]:
                exp[random_loc][2] = self.para_space[random_factor][1][1]
            elif exp[random_loc][2]>=self.para_space[random_factor][1][-1]:
                exp[random_loc][2] = self.para_space[random_factor][1][-2]
            elif np.random.rand()>0.5:
                exp[random_loc][2] = [i for i in self.para_space[random_factor][1] if i>exp[random_loc][2]][0]
            else:
                exp[random_loc][2] = [i for i in self.para_space[random_factor][1] if i<exp[random_loc][2]][-1]
        new = self.popu.type(exp)
        self.popu.add(new.code)
        return {new.code}
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
    def mutation_pool_and(self):
        pool0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = pool0.exp
        random_pop = np.random.randint(len(exp))
        if (np.random.rand()>0.5)&(len(exp)!=1):
            exp.pop(random_pop)
        else:
            random_factor = choice(self.basket)
            # 随机赋一个已有因子的权重
            exp.append(['equal' if self.para_space[random_factor][0] else\
                         'less' if np.random.rand()>0.5 else 'greater',\
                         random_factor, \
                         choice(self.para_space[random_factor][1])])
        new = self.popu.type(exp)
        self.popu.add(new.code)
        return {new.code}
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
    def mutation_pool_replace(self):
        pool0 = self.popu.type(list(self.popu.subset().codes)[0])
        exp = pool0.exp
        random_pop = np.random.randint(len(exp))
        # 删一个加一个
        if (len(exp)!=1):
            exp.pop(random_pop)
            random_factor = choice(self.basket)
            # 随机赋一个已有因子的权重
            exp.append(['equal' if self.para_space[random_factor][0] else\
                         'less' if np.random.rand()>0.5 else 'greater',\
                         random_factor, \
                         choice(self.para_space[random_factor][1])]) 
        new = self.popu.type(exp)
        self.popu.add(new.code)
        return {new.code}
    # 两因子交叉
    def cross_exchange(self):
        if len(self.popu.codes)<2:
            #print('种群规模过小')
            return {} 
        sele = list(self.popu.subset(2).codes)
        exp0 = self.popu.type(sele[0]).exp
        exp1 = self.popu.type(sele[1]).exp
        # 需要打乱顺序，保证交叉的多样性
        shuffle(exp0)
        shuffle(exp1)
        # 如果有一个是单因子的话则合成一个因子
        if (len(exp0)==1)|(len(exp1)==1):
            new = self.popu.type(exp0+exp1)
            self.popu.add(new.code)
            return {new.code}
        cut0 = np.random.randint(1,len(exp0))
        cut1 = np.random.randint(1,len(exp1))
        new_0 = self.popu.type(exp0[:cut0]+exp1[:cut1])
        new_1 = self.popu.type(exp0[cut0:]+exp1[cut1:])
        self.popu.add({new_0.code, new_1.code})
        return {new_0.code, new_1.code} 
    def get_seeds(self):
        if self.popu.type==(ind.Score):
            popu0 = popu.Population() 
            # 遍历单因子、双因子、三因子, 作为种子组合
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
        elif self.popu.type==(ind.Pool):
            popu0 = popu.Population(ind.Pool)
            # 单因子组合
            for factor in self.basket:
                for threshold in self.para_space[factor][1]:
                    if self.para_space[factor][0]:
                        pool0 = ind.Pool([['equal', factor, threshold]])
                        popu0.add(pool0.code)
                    else:
                        pool0 = ind.Pool([['less', factor, threshold]])
                        popu0.add(pool0.code)
                        pool0 = ind.Pool([['greater', factor, threshold]])
                        popu0.add(pool0.code)
            # 两因子组合
            popu0.add({ind.Pool(ind.Pool(i[0]).exp + ind.Pool(i[1]).exp).code\
                           for i in combinations(popu0.codes, 2)})
        return popu0.codes 
    # 种群繁殖
    def multiply(self, multi=2, prob_dict={}):
        # 各算子被执行的概率，如果空则全部算子等概率执形
        if prob_dict=={}:
            if self.popu.type==ind.Score:
                opts_mutation = [f for f in dir(Gen) if ('mutation' in f) and ('score' in f)]
            elif self.popu.type==ind.Pool:
                opts_mutation = [f for f in dir(Gen) if ('mutation' in f) and ('pool' in f)]
            opts_cross = [f for f in dir(Gen) if  'cross' in f]
            opts = opts_cross+opts_mutation
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
                #print(func)
                getattr(self, func)()


