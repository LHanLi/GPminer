import numpy as np
import pandas as pd
from random import shuffle,choice,sample
import GPminer as GPm
from itertools import combinations
import time, copy

# 种群繁殖类

class Gen():
    # 因子库（针对score和pool可以单独指定因子库），种群，市场，种群的类型
    def __init__(self, basket=[], score_basket=None, pool_basket=None,\
                  popu0=None, market=None, indtype='Score'):
        if (score_basket==None)&(pool_basket==None):
            self.score_basket = self.pool_basket = self.basket = basket
        elif score_basket==None:
            self.score_basket = self.basket = basket
            self.pool_basket = pool_basket
        elif pool_basket==None:
            self.pool_basket = self.basket = basket
            self.score_basket = score_basket
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
    def get_seeds(self, exclude=True):
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
        # 仅生成排除因子
        def seeds_Pool():
            popu0 = GPm.popu.Population(GPm.ind.Pool)
            # 单因子组合
            for factor in self.basket:
                for threshold in self.para_space[factor][1]:
                    # 离散变量使用=
                    if self.para_space[factor][0]:
                        if exclude:
                            pool0 = GPm.ind.Pool([[], [['equal', factor, [threshold, ]]]])
                        else:
                            pool0 = GPm.ind.Pool([[['equal', factor, [threshold, ]]], []])
                        popu0.add(pool0.code)
                    else:
                        if exclude:
                            pool0 = GPm.ind.Pool([[], [['less', factor, threshold]]])
                        else:
                            pool0 = GPm.ind.Pool([[['less', factor, threshold]], []])
                        popu0.add(pool0.code)
                        if exclude:
                            pool0 = GPm.ind.Pool([[], [['greater', factor, threshold]]])
                        else:
                            pool0 = GPm.ind.Pool([[['greater', factor, threshold]], []])
                        popu0.add(pool0.code)
            popu0 = popu0.subset(int(0.1*len(popu0.codes)))
            # 两因子组合
            if exclude:
                combos = [GPm.ind.Pool(i[0] + '|' + i[1][1:]).code \
                                 for i in combinations(popu0.codes, 2)]
            else:
                combos = [GPm.ind.Pool(i[0][:-1] + '|' + i[1]).code \
                                 for i in combinations(popu0.codes, 2)]
            popu0.add(set(sample(combos, int(0.1*len(combos)))))
            popu0.add(set(combos))
            return popu0
        if self.popu.type==(GPm.ind.Score):
            return seeds_Score().codes
        elif self.popu.type==(GPm.ind.Pool):
            return seeds_Pool().codes
        elif self.popu.type==(GPm.ind.SP):
            return set(i[0]+'&'+i[1] for i in zip(seeds_Score().codes, seeds_Pool().codes))
    # 增大或减小某因子参数
    def mutation_d(self, ind):
        exp = copy.deepcopy(ind.exp)
        if type(ind)==GPm.ind.Score:
            # 单因子权重无法改变
            if len(exp)==1:
                return ind
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
            new = GPm.ind.Score(exp)
            return new
        elif type(ind)==GPm.ind.Pool:
            # 等概率选择变异include或exclude部分（除非他们为空）
            if (len(exp[0])!=0)&((np.random.rand()<0.5)|(len(exp[1])==0)):
                select_inexlude = 0
                select_loc = np.random.randint(len(exp[0]))
            else:
                select_inexlude = 1
                select_loc = np.random.randint(len(exp[1]))
            thisfactor_space = self.para_space[exp[select_inexlude][select_loc][1]]
            # 对于离散型因子增加或减少value
            if thisfactor_space[0]:
                if len(exp[select_inexlude][select_loc][2])==1:
                    exp[select_inexlude][select_loc][2].append(choice(thisfactor_space[1]))
                else:
                    if np.random.rand()<0.5:
                        exp[select_inexlude][select_loc][2].append(choice(thisfactor_space[1]))
                    else:
                        exp[select_inexlude][select_loc][2].pop()
            # 对于数值型因子改变value到临近值
            else:
                less_value = sorted([i for i in thisfactor_space[1] if i<exp[select_inexlude][select_loc][2]])
                larger_value = sorted([i for i in thisfactor_space[1] if i>exp[select_inexlude][select_loc][2]])
                if (less_value == []) & (larger_value == []):
                    print('没有可变异的值')
                elif less_value==[]:
                    exp[select_inexlude][select_loc][2] = larger_value[0]
                elif larger_value==[]:
                    exp[select_inexlude][select_loc][2] = less_value[-1]
                else:
                    if np.random.rand()<0.5:
                        exp[select_inexlude][select_loc][2] = larger_value[0]
                    else:
                        exp[select_inexlude][select_loc][2] = less_value[-1]
            new = GPm.ind.Pool(exp)
            return new
        elif type(ind)==GPm.ind.SP:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<0.5:
                ind.score = self.mutation_d(ind.score)
            else:
                ind.pool = self.mutation_d(ind.pool)
            return GPm.ind.SP(ind.score.code+'&'+ind.pool.code)
        return 'pass all' 
    def popu_mutation_d(self):
        # 随机取出一个个体，变异得到新个体，添加得到个体。
        ind = self.popu.subset()
        ind = self.mutation_d(ind)
        self.popu.add(ind.code)
    # 增减因子
    def mutation_and(self, ind):
        exp = copy.deepcopy(ind.exp)
        if type(ind)==GPm.ind.Score:
            random_select0 = np.random.randint(len(exp))
            if (np.random.rand()>0.5)&(len(exp)!=1):
                exp.pop(random_select0)
            else:
                random_select = np.random.randint(len(self.basket))
                # 随机赋一个已有因子的权重
                exp.append([self.basket[random_select], np.random.rand()>0.5, \
                             exp[random_select0][2]])
            new = GPm.ind.Score(exp)
            return new
        elif type(ind)==GPm.ind.Pool:
            def expand(exp):
                exp = copy.deepcopy(exp)
                if (np.random.rand()>0.5)&(len(exp)!=1):
                    exp.pop()
                else:
                    random_factor = choice(self.basket)
                    exp.append(['equal' if self.para_space[random_factor][0] else\
                             'less' if np.random.rand()>0.5 else 'greater', random_factor, \
                             choice(self.para_space[random_factor][1])])
                return exp
            if exp[0]==[]:
                exp = [[], expand(exp[1])]
            elif exp[1]==[]:
                exp = [expand(exp[0]), []]
            else:
                if np.random.rand()<0.5:
                    exp = [expand(exp[0]), exp[1]]
                else:
                    exp = [exp[0], expand(exp[1])]
            new = GPm.ind.Pool(exp)
            return new
        elif type(ind)==GPm.ind.SP:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<0.5:
                ind.score = self.mutation_and(ind.score)
            else:
                ind.pool = self.mutation_and(ind.pool)
            return GPm.ind.SP(ind.score.code+'&'+ind.pool.code)
    def popu_mutation_and(self):
        ind = self.popu.subset()
        ind = self.mutation_and(ind)
        self.popu.add(ind.code)
    # 替换因子
    def mutation_replace(self, ind):
        exp = copy.deepcopy(ind.exp)
        if type(ind)==GPm.ind.Score:
            random_select0 = np.random.randint(len(exp))
            random_select = np.random.randint(len(self.basket))
            already = [i[0] for i in exp]
            # 不能替换前后相同，不能替换已有因子
            while (exp[random_select0][0]==self.basket[random_select])|\
                            (self.basket[random_select] in already):
                random_select = np.random.randint(len(self.basket))
            exp[random_select0][0] = self.basket[random_select]
            return GPm.ind.Score(exp)
        elif type(ind)==GPm.ind.Pool:
            # 删一个加一个
            def expreplace(exp):
                exp.pop()
                random_factor = choice(self.basket)
                # 随机赋一个已有因子的权重
                exp.append(['equal' if self.para_space[random_factor][0] else\
                                 'less' if np.random.rand()>0.5 else 'greater',\
                                 random_factor, \
                                 [choice(self.para_space[random_factor][1])]]) 
                return exp
            if exp[0]==[]:
                exp = [[], expreplace(exp[1])]
            elif exp[1]==[]:
                exp = [expreplace(exp[0]), []]
            else:
                if np.random.rand()<0.5:
                    exp = [[], expreplace(exp[1])]
                else:
                    exp = [expreplace(exp[1]), []]
            return GPm.ind.Pool(exp)
        else:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<0.5:
                ind.score = self.mutation_replace(ind.score)
            else:
                ind.pool = self.mutation_replace(ind.pool)
            return GPm.ind.SP(ind.score.code+'&'+ind.pool.code)
    def popu_mutation_replace(self):
        ind = self.popu.subset()
        ind = self.mutation_replace(ind)
        self.popu.add(ind.code)
    ## 两因子交叉
    #def cross_score_exchange(self):
    #    if len(self.popu.codes)<2:
    #        #GPm.ino.log('种群规模过小')
    #        return {} 
    #    sele = list(self.popu.subset(2).codes)
    #    exp0 = self.popu.type(sele[0]).exp
    #    exp1 = self.popu.type(sele[1]).exp
    #    # 需要打乱顺序，保证交叉的多样性
    #    shuffle(exp0)
    #    shuffle(exp1)
    #    # 如果有一个是单因子的话则合成一个因子
    #    if (len(exp0)==1)|(len(exp1)==1):
    #        new = self.popu.type(exp0+exp1)
    #        self.popu.add(new.code)
    #        return {new.code}
    #    cut0 = np.random.randint(1,len(exp0))
    #    cut1 = np.random.randint(1,len(exp1))
    #    new_0 = self.popu.type(exp0[:cut0]+exp1[:cut1])
    #    new_1 = self.popu.type(exp0[cut0:]+exp1[cut1:])
    #    self.popu.add({new_0.code, new_1.code})
    #    return {new_0.code, new_1.code} 
    # 种群繁殖
    def multiply(self, multi=2, prob_dict={}):
        # 各算子被执行的概率，如果空则全部算子等概率执形
        if prob_dict=={}:
            #if self.popu.type==GPm.ind.Score:
            #    opts = [f for f in dir(Gen) if  ('score' in f)]
            #elif self.popu.type==GPm.ind.Pool:
            #    opts = [f for f in dir(Gen) if  ('pool' in f)]
            #opts_cross = [f for f in dir(Gen) if  'cross' in f]
            opts = [f for f in dir(Gen) if  ('popu' in f)]
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


