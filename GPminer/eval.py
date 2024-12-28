import pandas as pd
import numpy as np
import FreeBack as FB
import GPminer as GPm 
import time

# 计算个体适应度类

class Eval():
    def __init__(self, market, pool=None, score=None):
        self.market = market
        self.pool = pool
        self.score = score
    def eval_pool(self, poolcode=None, mod='or'):
        if poolcode!=None:
            if mod=='or':
                self.pool = GPm.ind.Pool(poolcode)
            elif mod=='and':
                self.pool = GPm.ind.Pooland(poolcode)
        # 如果pool为None则include=True
        if self.pool!=None:
            # 默认全包含
            if self.pool.exp[0]!=[]:
                result = []
                for c in self.pool.exp[0]:
                    if c[0]=='less':
                        r=(self.market[c[1]]<c[2])
                    elif c[0]=='greater':
                        r=(self.market[c[1]]>c[2])
                    elif c[0]=='equal':
                        r=(self.market[c[1]].isin(c[2]))
                    result.append(r)
                if mod=='or':
                    include = pd.concat(result, axis=1).any(axis=1)
                elif mod=='and':
                    include = pd.concat(result, axis=1).all(axis=1)
            else:
                include = pd.Series(True, index=self.market.index)
            # 默认不排除
            if self.pool.exp[1]!=[]:
                result = []
                for c in self.pool.exp[1]:
                    if c[0]=='less':
                        r=(self.market[c[1]]<c[2])
                    elif c[0]=='greater':
                        r=(self.market[c[1]]>c[2])
                    elif c[0]=='equal':
                        r=(self.market[c[1]].isin(c[2]))
                    result.append(r)
                if mod=='or':
                    exclude = pd.concat(result, axis=1).any(axis=1)
                elif mod=='and':
                    exclude = pd.concat(result, axis=1).all(axis=1)
            else:
                exclude = pd.Series(False, index=self.market.index)
            self.market['include'] = include&(~exclude)
        else:
            self.market['include'] = True 
    def eval_score(self, scorecode=None):
        #time0 = time.time()
        if scorecode!=None:
            self.score = GPm.ind.Score(scorecode)
        # 获取筛选/排除后factor排序
        def process_factor(factor_name):
            if self.score.rankall:
                return self.market[factor_name]
            else:
                # 排除掉的置为np.nan
                return self.market[factor_name]*(self.market['include'].astype(int).\
                                                replace(to_replace={0:np.nan}))
        # 获取打分
        for factor in self.score.exp:
            self.market[factor[0]+'_score'] = process_factor(factor[0]).groupby('date').\
                rank(ascending=factor[1])*factor[2]
        basescore = [i[0]+'_score' for i in self.score.exp]
        self.market['score'] = self.market[basescore].sum(axis=1)
        #ino.log('获取打分耗时', time.time()-time0)
        #time0 = time.time()
    def backtest(self, hold_num, price, code_returns=None, interval=1):
        # 回测
        strat0 = FB.strat.MetaStrat(self.market, 'include', 'score',\
                    hold_num, price, interval=interval, code_returns=code_returns)
        strat0.run()
        return strat0


