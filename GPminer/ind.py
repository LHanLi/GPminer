import math
from random import shuffle

# 种群的个体单元


# 打分因子(code:'2*False*a+1*True*b, exp:[[2, False, 'a'], [1, True, 'b']])
class Score():
    max_exp_len = 5 # 最大因子数
    max_mul = 10 # 最大因子系数
    rankall = False # 池子外股票是否参与排序
    # 支持编码和表达式两种形式生成打分因子
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
    # 比较打分因子大小
    def compare(self, s0):
        # 先比较因子数量
        ns = s0.count('+')
        # 再比较权重数字大小
        ws = sum([float(i.split('*')[0]) for i in s0.split('+')]) 
        return 1e5*ns+ws 

# 排除/选取因子（确定策略池子） 
# (code:'a<130|b='A'), exp:[['less', 'a', 130], ['equal', 'b', 'A']])
class Pool():
    inexclude = 'exclude'
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
    def code2exp(self):
        exp = []
        split = self.code.split('|')
        for i in split:
            # 组合中的单个条件
            if '<' in i:
                opt = 'less'
                s = i.split('<')
                try:
                    value = float(s[1])
                except:
                    value = s[1]
                factor = s[0]
            elif '>' in i:
                opt='greater'
                s = i.split('>')
                try:
                    value = float(s[1])
                except:
                    value = s[1]
                factor = s[0]
            elif '=' in i:
                opt='equal'
                s = i.split('=')
                try:
                    value = float(s[1])
                except:
                    value = s[1]
                factor = s[0]
            one = [opt, factor, value] 
            exp.append(one)
        self.exp = exp
    def exp2code(self):
        self.code = '|'.join([i[1]+(lambda x:'<' if x=='less' else '>' if x=='greater' \
                               else '=' if x=='equal' else 'unknown')\
                                (i[0])+str(i[2]) for i in self.exp])
    def uexp(self):
        # 先按因子名称排序，再按逻辑符号，再按值
        def takewsort(one):
            return one[1:2]+one[:1]+one[2:]
        exp = self.exp
        exp.sort(key=takewsort, reverse=True)
        # 大小于号重叠部分去除，等于重复去除
        def unique_c(exp):
            prefactor = ''
            preopt = ''
            unique_exp = []
            for c in exp:
                opt = c[0]
                factor = c[1]
                if (factor==prefactor)&(opt==preopt):
                    if c[0]=='less':
                        if c[2]>min(values):
                            unique_exp.pop()
                            unique_exp.append(c)
                    elif c[0]=='equal':
                        if c[2] not in values:
                            unique_exp.append(c)
                    elif c[0]=='greater':
                        if c[2]<max(values):
                            unique_exp.pop()
                            unique_exp.append(c)
                    values.append(c[2])
                else:
                    #print('不同因子或算子直接跳过')
                    unique_exp.append(c)
                    prefactor = factor
                    preopt = opt
                    values = [c[2]]
                    continue
            return unique_exp
        self.exp = unique_c(exp)
    def factors(self):
        return set([i[1] for i in self.exp])