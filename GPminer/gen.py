import numpy as np
import pandas as pd
from random import shuffle,choice,sample
import GPminer as GPm
from itertools import combinations
import time, copy

# 种群繁殖类

class Gen():
    max_seeds = 10000
    # 因子库（针对score和pool可以单独指定因子库），种群，市场，种群的类型 打分排除因子变异比例(1表示只变异打分因子，0表示只变异排除因子)
    def __init__(self, popu0=None, market=None, indtype=GPm.ind.Score,\
                        score_basket=[], pool_basket=[], mutation_ratio=0.5):
        #if (type(score_basket)==type(None))&(type(pool_basket)==type(None)):
        #    self.score_basket = self.pool_basket = self.basket = basket
        #elif type(score_basket)==type(None):
        #    self.score_basket = self.basket = basket
        #    self.pool_basket = pool_basket
        #elif type(pool_basket)==type(None):
        #    self.pool_basket = self.basket = basket
        #    self.score_basket = score_basket
        #else:
        #    self.score_basket = score_basket
        #    self.pool_basket = pool_basket
        self.score_basket = score_basket
        self.pool_basket = pool_basket
        if popu0==None:
            self.popu = GPm.popu.Population(indtype)
        else:
            self.popu = popu0
        #if  self.popu.type != GPm.ind.Score:
            #(self.popu.type==GPm.ind.Pooland) | (self.popu.type==GPm.ind.Pool) |\
            #(self.popu.type==GPm.ind.SP):
        # 初始化因子的参数域，需要输入market
        if type(market)==type(None):
            print('market is needed for Pool ind Gen!')
            return 
        if indtype!=GPm.ind.Score:
            self.para_space = {}
            for factor in list(set(self.pool_basket)|set(self.score_basket)):
                # 数值因子，小于等于divide_n个数时全部因子值进入参数空间
                divide_n = 100
                if type(market[factor].iloc[0]) in \
                    [np.float32, np.float64, np.int64, type(1.0), type(1)]:
                    if len(market[factor].unique())>divide_n:
                        self.para_space[factor] = (False, [market[factor].quantile(i) \
                                    for i in np.linspace(0.01,0.99,divide_n)]) 
                    else:  # 最大最小值去除
                        self.para_space[factor] = (False, sorted(market[factor].unique())[1:-1]) 
                else:
                    self.para_space[factor] = (True, list(market[factor].unique())) 
            # 打分因子只能是数值型因子
            self.score_basket = [i for i in self.score_basket if not self.para_space[i][0]] 
        else:
            self.score_basket = list(market[self.score_basket].select_dtypes(include=['number']).columns)
        GPm.ino.log('非数值型因子无法作为打分因子,同时得到选股因子的阈值空间,最终得到%s个打分因子'\
                    %(len(self.score_basket)))
        self.mutation_ratio = mutation_ratio
    # 从basket中因子获得popu
    def get_seeds(self, exclude=True):
        GPm.ino.log('最大种子数量%s'%self.max_seeds)
        def seeds_Score(max_seeds):
            popu0 = GPm.popu.Population() 
            allseeds = ['1*'+i+'*'+j for i in ['True', 'False'] for j in self.score_basket]
            if len(allseeds)>=max_seeds:
                seeds = sample(allseeds, max_seeds)
                popu0.add(set(seeds))
                GPm.ino.log('生成单因子种子%s个,选取%s个作为种子'%(len(allseeds), max_seeds))
            else:   # 如果单因子种子不够则再增加双因子组合
                popu0.add(set(allseeds))
                GPm.ino.log('单因子数量为%s不足%s,增加双因子组合'%(len(allseeds), max_seeds))
                allseeds = ['1*%s*'%a+i+'+'+'1*%s*'%b+j for i,j in \
                    combinations(self.score_basket) \
                        for a in ['True', 'False'] for b in ['True', 'False']]
                allseeds = [GPm.ind.Score(s).code for s in allseeds]
                GPm.ino.log('单因子种子数量不足,生成双因子种子%s个'%len(allseeds))
                seeds = sample(allseeds, max_seeds-len(popu0.codes))
                popu0.add(set(seeds))  # 已经提前排序,直接添加即可
                #for s in seeds:     # 多因子需要调整顺序，得到唯一字符串
                #    popu0.add(GPm.ind.Score(s).code)
            GPm.ino.log('生成%s Score种子'%len(popu0.codes))
            return popu0
        # 仅生成排除因子
        def seeds_Pool(max_seeds):
            popu0 = GPm.popu.Population(GPm.ind.Pool)
            allseeds = []
            for factor in self.pool_basket:   # 单因子组合
                for threshold in self.para_space[factor][1]:
                    # 离散变量使用=
                    if self.para_space[factor][0]:
                        if exclude:
                            s = ';%s=%s'%(factor, threshold)
                        else:
                            s = '%s=%s;'%(factor, threshold)
                        allseeds.append(s)
                    else:
                        if threshold!=self.para_space[factor][1][0]: # 最小的阈值不需要小于算子
                            if exclude:
                                s = ';%s<%s'%(factor, threshold)
                            else:
                                s = '%s<%s;'%(factor, threshold)
                            allseeds.append(s)
                        if threshold!=self.para_space[factor][1][-1]:
                            if exclude:
                                s = ';%s>%s'%(factor, threshold)
                            else:
                                s = '%s>%s;'%(factor, threshold)
                            allseeds.append(s)
            if max_seeds<=len(allseeds):
                seeds = sample(allseeds, max_seeds)
            else:
                seeds = allseeds
                if exclude:
                    allseeds = [GPm.ind.Pool(i+'|'+j[1:]) for i,j in combinations(allseeds, 2)]
                    allseeds = [i.code for i in allseeds if len(i.exp[1])>1]
                else:
                    allseeds = [GPm.ind.Pool(i[:-1]+'|'+j) for i,j in combinations(allseeds, 2)]
                    allseeds = [i.code for i in allseeds if len(i.exp[0])>1]
                seeds = seeds + sample(allseeds, self.max_seeds-len(seeds))
            popu0.add(set(seeds))
            GPm.ino.log('生成%s Pool种子'%len(popu0.codes))
            return popu0 
        if self.popu.type==(GPm.ind.Score):
            return seeds_Score(self.max_seeds).codes
        # Pool和Pooland的code/exp是互通的
        elif  (self.popu.type==GPm.ind.Pooland) | (self.popu.type==(GPm.ind.Pool)):
            return seeds_Pool(self.max_seeds).codes
        elif self.popu.type==(GPm.ind.SP):
            seeds_score = seeds_Score(int(np.sqrt(self.max_seeds))+10)
            seeds_pool = seeds_Pool(int(np.sqrt(self.max_seeds))+10)
            mix = [i+'&'+j for i in seeds_score.codes for j in seeds_pool.codes]
            GPm.ino.log('生成%s SP种子'%len(mix))
            return set(sample(mix, self.max_seeds))
    # 增大或减小某因子参数
    def mutation_d(self, ind):
        exp = copy.deepcopy(ind.exp)
        if type(ind)==GPm.ind.Score:
            # 单因子权重无法改变
            if len(exp)==1:
                return ind
            random_select = np.random.randint(len(exp)) 
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
        elif  (type(ind)==GPm.ind.Pooland) | (type(ind)==GPm.ind.Pool):
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
                    #print('没有可变异的值')
                    pass
                elif less_value==[]:
                    exp[select_inexlude][select_loc][2] = larger_value[0]
                elif larger_value==[]:
                    exp[select_inexlude][select_loc][2] = less_value[-1]
                else:
                    if np.random.rand()<0.5:
                        exp[select_inexlude][select_loc][2] = larger_value[0]
                    else:
                        exp[select_inexlude][select_loc][2] = less_value[-1]
            new = type(ind)(exp)
            return new
        elif type(ind)==GPm.ind.SP:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<self.mutation_ratio:
                score = self.mutation_d(ind.score)
                return GPm.ind.SP(score.code+'&'+ind.pool.code)
            else:
                pool = self.mutation_d(ind.pool)
                return GPm.ind.SP(ind.score.code+'&'+pool.code)
        return 'pass all' 
    def popu_mutation_d(self):
        # 随机取出一个个体，变异得到新个体，添加得到个体。
        ind = self.popu.subset()
        ind = self.mutation_d(ind)
        self.popu.add(ind.code)
    # 增减因子
    def mutation_and(self, ind):
        exp = copy.deepcopy(ind.exp)
        #GPm.ino.log('mutation_and 变异%s'%(ind.code))
        if type(ind)==GPm.ind.Score:
            random_select0 = np.random.randint(len(exp))
            if (np.random.rand()>0.5)&(len(exp)!=1):
                exp.pop(random_select0)
            else:
                random_select = np.random.randint(len(self.score_basket))
                # 随机赋一个已有因子的权重
                exp.append([self.score_basket[random_select], np.random.rand()>0.5, \
                             exp[random_select0][2]])
            new = GPm.ind.Score(exp)
            return new
        elif (type(ind)==GPm.ind.Pooland) | (type(ind)==GPm.ind.Pool):
            def expand(exp):
                exp = copy.deepcopy(exp)
                if (np.random.rand()>0.5)&(len(exp)!=1):
                    exp.pop()
                else:
                    random_factor = choice(self.pool_basket)
                    if self.para_space[random_factor][0]:
                        exp.append(['equal', random_factor, [choice(self.para_space[random_factor][1])]])
                    else:
                        exp.append(['less' if np.random.rand()>0.5 else 'greater', random_factor, \
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
            new = self.popu.type(exp)
            return new
        elif type(ind)==GPm.ind.SP:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<self.mutation_ratio:
                score = self.mutation_and(ind.score)
                return GPm.ind.SP(score.code+'&'+ind.pool.code)
            else:
                pool = self.mutation_and(ind.pool)
                return GPm.ind.SP(ind.score.code+'&'+pool.code)
    def popu_mutation_and(self):
        ind = self.popu.subset()
        ind = self.mutation_and(ind)
        self.popu.add(ind.code)
    # 替换因子
    def mutation_replace(self, ind):
        exp = copy.deepcopy(ind.exp)
        if type(ind)==GPm.ind.Score:
            random_select0 = np.random.randint(len(exp))
            random_select = np.random.randint(len(self.score_basket))
            already = [i[0] for i in exp]
            # 不能替换前后相同，不能替换已有因子
            while (exp[random_select0][0]==self.score_basket[random_select])|\
                            (self.score_basket[random_select] in already):
                random_select = np.random.randint(len(self.score_basket))
            exp[random_select0][0] = self.score_basket[random_select]
            return GPm.ind.Score(exp)
        elif (type(ind)==GPm.ind.Pooland) | (type(ind)==GPm.ind.Pool):
            # 删一个加一个
            def expreplace(exp):
                exp.pop()
                random_factor = choice(self.pool_basket)
                # 随机赋一个para_space中阈值
                if self.para_space[random_factor][0]: 
                    exp.append(['equal', random_factor, \
                                    [choice(self.para_space[random_factor][1])]])
                else:
                    exp.append(['less' if np.random.rand()>0.5 else 'greater',\
                                 random_factor, choice(self.para_space[random_factor][1])])
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
            return type(ind)(exp)
        else:
            # 随机选打分因子/排除因子变异
            if np.random.rand()<self.mutation_ratio:
                score = self.mutation_replace(ind.score)
                return GPm.ind.SP(score.code+'&'+ind.pool.code)
            else:
                pool = self.mutation_replace(ind.pool)
                return GPm.ind.SP(ind.score.code+'&'+pool.code)
    def popu_mutation_replace(self):
        ind = self.popu.subset()
        ind = self.mutation_replace(ind)
        self.popu.add(ind.code)
    # 合成因子
    def mutation_sum(self, ind0, ind1):
        #GPm.ino.log('mutation_sum 变异%s, %s'%(ind0.code, ind1.code))
        if type(ind0)==GPm.ind.Score:
            return GPm.ind.Score(ind0.code+'+'+ind1.code)
        elif  (type(ind0)==GPm.ind.Pooland) | (type(ind0)==GPm.ind.Pool):
            def split(ind, exclude=True):
                indsplit = ind.code.split(';')
                if exclude:
                    return indsplit[1:] if indsplit[1]!='' else []
                else:
                    return indsplit[:1] if indsplit[0]!='' else []
            ind = '|'.join(split(ind0, False)+split(ind1, False))+';'+\
                                '|'.join(split(ind0)+split(ind1))
            return GPm.ind.Pool(ind)
        else:
            return self.mutation_sum(ind0.score, ind1.score).code+'&'+\
                        self.mutation_sum(ind0.pool, ind1.pool)
    def popu_mutation_sum(self):
        ind = self.mutation_sum(self.popu.subset(), self.popu.subset())
        self.popu.add(ind.code)
    # 种群繁殖
    def multiply(self, multi=2, prob_dict={}):
        # 各算子被执行的概率，如果空则全部算子等概率执形
        if prob_dict=={}:
            opts = [f for f in dir(Gen) if  ('popu' in f)]
            prob_ser = pd.Series(np.ones(len(opts)), index=opts)
        else:
            prob_ser = pd.Series(prob_dict.values(), index=prob_dict.keys())
        prob_ser = prob_ser/prob_ser.sum()
        prob_ser = prob_ser.cumsum()
        # 种群繁殖到目标数量，同时限制单次变异最大时间5s
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
                pass
                #GPm.ino.log('warning!!! %s超过最大运行时间5s'%func)


