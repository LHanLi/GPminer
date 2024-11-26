import math
import numpy as np 
import pandas as pd
from random import shuffle

# 种群的个体单元
# 可以通过code或exp生成，code和exp一一对应，code用于hash，exp用于计算。
class Ind():
    def __init__(self, input=None):
        if type(input)==type(""):
            self.code = input
            self.code2exp()
            self.uexp()
            self.exp2code()
        elif type(input)==type([]):
            self.exp = input
            self.uexp()
            self.exp2code()
        else:
            self.exp = None
            self.code = None
    # 从code获得exp
    def code2exp(self):
        pass
    # 从exp获得code
    def exp2code(self):
        pass
    # 保证表达式唯一性
    def uexp(self):
        pass

# 打分因子, 
# code:'2*False*a+1*True*b, exp:[[2, False, 'a'], [1, True, 'b']]
# 2倍a因子越小越好+1倍b因子越大越好
class Score(Ind):
    max_exp_len = 10 # 最大因子数
    max_mul = 50 # 最大因子系数          constant
    rankall = False # 池子外股票是否参与排序
    def code2exp(self):
        exp = []
        split = self.code.split('+')
        for i in split:
            # 组合中的单个因子
            one = i.split('*')
            one = [one[2], one[1]=='True', int(one[0])]
            exp.append(one)
        self.exp = exp
    def exp2code(self):
        self.code = '+'.join([str(int(i[2]))+'*'+str(i[1])+'*'+i[0] for i in self.exp])
    # 保证等价的表达式唯一
    def uexp(self):
        exp = [] 
        already = []
        shuffle(self.exp)  # 当因子重复出现时添加一些随机性
        for i in self.exp:
            # 如果出现0或负值则直接跳过
            if i[2]<=0:
                continue
            # 如果同一因子出现两次以上则只保留第一个
            elif i[0] not in already:
                already.append(i[0])
                exp.append(i)
        exp = exp[:self.max_exp_len]
        # 除以最大公因数
        ws = [i[2] for i in exp] 
        smaller=int(min(ws))
        for i in reversed(range(1,smaller+1)):
            if list(filter(lambda j: j%i!=0,ws)) == []:
                hcf = i
                break
        # 限制最大同时考虑等权情况
        exp = [[i[0], i[1], min(i[2]/hcf, self.max_mul)] for i in exp]
        if max([i[2] for i in exp])==min([i[2] for i in exp]):
            exp = [[i[0], i[1], 1] for i in exp]
        # 先按权重排序，再按因子名称排序，再按方向排序
        def takewsort(one):
            return one[::-1]
        exp.sort(key=takewsort, reverse=True)
        self.exp = exp
    # 获取其包含的全部因子
    def factors(self):
        count = {}
        for i in self.exp:
            try:
                count[i[0]] += i[2]
            except:
                count[i[0]] = i[2]
        return pd.Series(count).sort_values(ascending=False)
    # 比较打分因子大小
    def compare(self, s0):
        # 先比较因子数量
        ns = s0.count('+')
        # 再比较权重数字大小
        ws = sum([float(i.split('*')[0]) for i in s0.split('+')]) 
        return 1e5*ns+ws 

# 排除/选取因子（确定策略池子）
# code：'a<130|b=A,B|c>C'  exp:[[['less', 'a', 130]], [['equal', 'b', ['A', 'B']], ['great', 'c', 'C']]
# ;前为include，后为exclude条件, 意为全部a<130的股票中排除掉b为A和B以及c大于C的股票。 
class Pool(Ind):
    def code2exp(self):
        innex = self.code.split(';')
        final_exp = []
        for code in innex:
            exp = []
            if code=='':
                final_exp.append(exp)
                continue
            split = code.split('|')
            for i in split:
                # 组合中的单个条件
                if '<' in i:
                    opt = 'less'
                    s = i.split('<')
                    value = float(s[1])
                    factor = s[0]
                elif '>' in i:
                    opt='greater'
                    s = i.split('>')
                    value = float(s[1])
                    factor = s[0]
                else:
                    opt='equal'
                    s = i.split('=')
                    value = []
                    for i in s[1].split(','):
                        try:
                            value.append(float(i))
                        except:
                            value.append(i)
                    factor = s[0]
                one = [opt, factor, value]
                exp.append(one)
            final_exp.append(exp)
        self.exp = final_exp
    def exp2code(self):
        code = []
        for exp in self.exp:
            code.append('|'.join([i[1]+\
                (lambda x:'<' if x=='less' else '>' if x=='greater' else '=')(i[0]) +\
                 (str(i[2]) if type(i[2])!=list else ','.join([str(j) for j in i[2]])) for i in exp]))
        self.code = ';'.join(code)
    def uexp(self):
        def unique_c(exp):
            # 先按因子名称排序，再按逻辑符号，再按值
            def takewsort(one):
                return [one[1], one[0], str(one[2])]
            exp.sort(key=takewsort, reverse=True)
            # 大小于号重叠部分去除，等于重复去除
            prefactor = ''
            preopt = ''
            prevalue = 0
            unique_exp = []
            for c in exp:
                opt = c[0]
                factor = c[1]
                value = c[2]
                #print('因子和操作符都相同时需要考虑合并问题')
                if (factor==prefactor)&(opt==preopt):
                    if opt=='equal':   # 新元素全部并入
                        unique_exp.pop()
                        unique_exp.append([opt, factor, sorted([str(i) for i in list(values|set(value))])])
                        values = values|set(value)
                        continue
                    elif opt=='less':
                        # 如果集合扩大则更新
                        if value>max(values):
                            unique_exp.pop()
                            unique_exp.append(c)
                    elif opt=='greater':
                        if value<min(values):
                            unique_exp.pop()
                            unique_exp.append(c)
                    values.append(value)
                else:
                    #print('因子相同，操作符不同时，可能表达式代表的是空集，如a<10|a>5则随机去掉一个条件)
                    if (factor==prefactor)&(opt!=preopt)&(opt!='equal'):
                        if (((opt=='less')&(value>=prevalue))|((opt=='greater')&(value<=prevalue))):
                            if np.random.rand()<0.5:
                                pass  # 去掉该条件
                            else:
                                unique_exp.pop()   # 去掉前一个条件
                                unique_exp.append(c)
                            continue
                    preopt = opt
                    prefactor = factor
                    prevalue = value
                    if opt=='equal':
                        if type(value)==list:
                            values = set(value)
                        else:
                            values = set([value])
                        unique_exp.append([opt, factor, sorted(list(values))])
                    else:
                        values = [value, ]
                        unique_exp.append(c)
            return unique_exp
        self.exp = [unique_c(self.exp[0]), unique_c(self.exp[1])]
    def factors(self):
        count = {}
        for i in self.exp:
            for j in i:
                try:
                    count[j[1]] += 1 
                except:
                    count[j[1]] = 1 
        return pd.Series(count).sort_values(ascending=False)
    
# 策略类，包含Score和Pool
# code: Score.code+'&'+Pool.code, exp: [Score.exp, Pool.exp]
class SP(Ind):
    def __init__(self, input=None):
        self.score = self.pool = None
        super().__init__(input)
    def code2exp(self):
        if self.score==self.pool==None:        
            scorecode, poolcode = self.code.split('&')
            self.score = Score(scorecode)
            self.pool = Pool(poolcode)
        self.exp = [self.score.exp, self.pool.exp]
    def exp2code(self):
        if self.score==self.pool==None:        
            scoreexp, poolexp = self.exp
            self.score = Score(scoreexp)
            self.pool = Pool(poolexp)
        self.code = self.score.code+'&'+self.pool.code
    def factors(self):
        return self.score.factors().add(self.pool.factors(), fill_value=0).sort_values(ascending=False)

