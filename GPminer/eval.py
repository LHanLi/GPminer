import pandas as pd
import numpy as np
import FreeBack as FB
from GPminer import * 
import time

# 计算个体适应度类

class Eval():
    hold_num = 5 # 持有hold_num只
    comm = 10/1e4 # 滑点与佣金
    price = 'close' # 结算价格
    def __init__(self, market, pool=None, score=None):
        self.market = market
        self.pool = pool
        self.score = score
        if self.pool!=None:
            # 默认全包含
            if self.pool.exp[0]!=[]:
                result = []
                for c in self.pool.exp[0]:
                    if c[0]=='less':
                        r=(market[c[1]]<c[2])
                    elif c[0]=='greater':
                        r=(market[c[1]]>c[2])
                    elif c[0]=='equal':
                        r=(market[c[1]]==c[2])
                    result.append(r)
                include = pd.concat(result, axis=1).any(axis=1)
            else:
                include = pd.Series(True, index=self.market.index)
            # 默认不排除
            if self.pool.exp[1]!=[]:
                result = []
                for c in self.pool.exp[1]:
                    if c[0]=='less':
                        r=(market[c[1]]<c[2])
                    elif c[0]=='greater':
                        r=(market[c[1]]>c[2])
                    elif c[0]=='equal':
                        r=(market[c[1]]==c[2])
                    result.append(r)
                exclude = pd.concat(result, axis=1).any(axis=1)
            else:
                exclude = pd.Series(False, index=self.market.index)
            self.market['include'] = include&(~exclude) 
    def eval_score(self, score0=None):
        #time0 = time.time()
        if score0!=None:
            self.score = ind.Score(score0)
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
        #print('获取打分耗时', time.time()-time0)
        #time0 = time.time()
    def backtest(self):
        # 回测
        strat0 = FB.strat.MetaStrat(self.market, 'include', 'score',\
                                     self.hold_num, self.price)
        strat0.run()
        #print('策略回测耗时', time.time()-time0)
        #time0 = time.time()
        # 后处理
        self.post = FB.post.StratPost(strat0, self.market, comm=self.comm, fast=True)
        #print('后处理耗时', time.time()-time0)
        #time0 = time.time()
        return self.post


