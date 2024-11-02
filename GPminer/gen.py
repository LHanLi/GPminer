import numpy as np
import pandas as pd
from random import shuffle,choice,sample
import GPminer as GPm
from itertools import combinations
import time

# 种群繁殖类

class Gen():
    # 因子库，种群，市场，种群的类型
    def __init__(self, basket=[], popu0=None, market=None, indtype='Score'):
        self.basket = basket
        if popu0==None:
            if indtype=='Score':
                self.popu = GPm.popu.Population(GPm.ind.Score)
            elif indtype=='Pool':
                self.popu = GPm.popu.Population(GPm.ind.Pool)
            elif indtype=='SP':
                self.popu = GPm.popu.Population(GPm.ind.SP)
        else:
            self.popu = popu0
        # 初始化pool的参数域，需要输入market
        if (self.popu.type==GPm.ind.Pool) | (self.popu.type==GPm.ind.SP):
            if type(market)==type(None):
                print('market is needed for Pool ind Gen!')
                return 
            self.para_space = {}
            for factor in basket:
                try:
                    self.para_space[factor] = (False, [market[factor].quantile(i) \
                                    for i in np.linspace(0.01,0.99,21)])   # 数值因子
                except:         # 离散因子
                    self.para_space[factor] = (True, list(market[factor].unique())) 
    # 从basket中因子获得popu
    def get_seeds(self):
        def seeds_Score():
            popu0 = GPm.popu.Population() 
            # 遍历单因子、双因子, 作为种子组合
            for i in self.basket:
                popu0.add({GPm.ind.Score([[i, True, 1]]).code, GPm.ind.Score([[i, False, 1]]).code})
            for i,j in list(combinations(self.basket, 2)):
                for b0 in [True, False]:
                    for b1 in [True, False]:
                        popu0.add(GPm.ind.Score([[i, b0, 1], [j, b1, 1]]).code)
            return popu0
        def seeds_Pool():
            popu0 = GPm.popu.Population(GPm.ind.Pool)
            # 单因子组合
            for factor in self.basket:
                for threshold in self.para_space[factor][1]:
                    # 离散变量使用=
                    if self.para_space[factor][0]:
                        pool0 = GPm.ind.Pool([[], [['equal', factor, threshold]]])
                        popu0.add(pool0.code)
                    else:
                        pool0 = GPm.ind.Pool([[], [['less', factor, threshold]]])
                        popu0.add(pool0.code)
                        pool0 = GPm.ind.Pool([[], [['greater', factor, threshold]]])
                        popu0.add(pool0.code)
            popu0 = popu0.subset(int(0.1*len(popu0.codes)))
            # 两因子组合
            combos = [GPm.ind.Pool(i[0] + '|' + i[1][1:]).code \
                                  for i in combinations(popu0.codes, 2)]
            popu0.add(set(sample(combos, int(0.1*len(combos)))))
            return popu0
        if self.popu.type==(GPm.ind.Score):
            return seeds_Score().codes
        elif self.popu.type==(GPm.ind.Pool):
            return seeds_Pool().codes
        elif self.popu.type==(GPm.ind.SP):
            return set(i[0]+'&'+i[1] for i in zip(seeds_Score().codes, seeds_Pool().codes))
    # 增大或减小某因子参数
    def mutation_d(self, ind):
        if type(ind)==GPm.ind.Score:
            exp = ind.exp
            # 单因子权重无法改变
            if len(exp)==1:
                return {ind.code}
            random_select = np.random.randint(len(exp)) 
            #print('选择变异第%s个因子权重'%random_select)
            deltawmax = 0.1 # 权重改变幅度小于此阈值   # 声明常数 constant
            deltawmin = 0.02 # 权重改变幅度大于此阈值
            max_step = 100 # 寻找新权重组合最大尝试次数
            # 全部权重乘mul，随机选一个因子权重+-d(\
            # 改变random_select的权重可以通过 1.改变该因子的乘数 或 2.改变其他因子的乘数实现)
            sumw = sum([i[2] for i in exp])
            wbefore = exp[random_select][2]/sumw  # 因子权重
            mul = d = 1  
            # 使用mehtod、opt、mul、d改变乘数后该因子权重
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
            # 增大、减小权重50%概率
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
                    # GPm.ino.log(mul, d)
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
                #GPm.ino.log('通过%s系数, 减小权重, mul=%s, d=%s'%\
                #      ((lambda x: '减小' if x else '增大')(method), mul, d))
            score_new = GPm.ind.Score(exp)
            return {score_new.code}
        elif type(ind)==GPm.ind.Pool:
            exp = ind.exp
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
            return {new.code}
    def mutation_score_d(self):
        # 随机取出一个个体，变异得到新个体，添加得到个体。
        score0 = self.popu.type(self.popu.subset().codes.pop())
        score_new = self.mutation_d(score0)
        self.popu.add(score_new.code)
    def mutation_pool_d(self):
        pool0 = self.popu.type(self.popu.subset().codes.pop())
        pool_new = self.mutation_d(pool0)
        self.popu.add(pool_new.code) 
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
        score_new = GPm.ind.Score(exp)
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
        score_new = GPm.ind.Score(exp)
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
    def cross_score_exchange(self):
        if len(self.popu.codes)<2:
            #GPm.ino.log('种群规模过小')
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
    # 种群繁殖
    def multiply(self, multi=2, prob_dict={}):
        # 各算子被执行的概率，如果空则全部算子等概率执形
        if prob_dict=={}:
            if self.popu.type==GPm.ind.Score:
                opts = [f for f in dir(Gen) if  ('score' in f)]
            elif self.popu.type==GPm.ind.Pool:
                opts = [f for f in dir(Gen) if  ('pool' in f)]
            #opts_cross = [f for f in dir(Gen) if  'cross' in f]
            prob_ser = pd.Series(np.ones(len(opts)), index=opts)
        else:
            prob_ser = pd.Series(prob_dict.values(), index=prob_dict.keys())
        prob_ser = prob_ser/prob_ser.sum()
        prob_ser = prob_ser.cumsum()
        # 种群繁殖到目标数量，同时限制单次变异最大时间
        popu_size = len(self.popu.codes)
        from func_timeout import func_set_timeout
        @func_set_timeout(5)
        def run_mul(func):
            getattr(self, func)()
        while len(self.popu.codes)<int(popu_size*multi):
            r = np.random.rand()
            #GPm.ino.log('算子选择随机数：%.3lf'%r)
            func = prob_ser[prob_ser>r].index[0]
            try:
                run_mul(func)
                #GPm.ino.log('执行完毕')
            except:
                GPm.ino.log('warning!!! %s超过最大运行时间5s'%func)


