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
    def __init__(self, market, pool0=None, score0=None):
        self.market = market
        self.pool = pool0
        self.score = score0
        if self.pool!=None:
            result = []
            for c in self.pool.exp:
                if c[0]=='less':
                    r=(market[c[1]]<c[2])
                elif c[0]=='greater':
                    r=(market[c[1]]>c[2])
                elif c[0]=='equal':
                    r=(market[c[1]]==c[2])
                result.append(r)
            self.market[self.pool.inexclude] = pd.concat(result, axis=1).any(axis=1)
    def eval_score(self, p):
        #time0 = time.time()
        self.score = ind.Score(p)
        # 获取筛选/排除后factor排序
        def process_factor(factor_name):
            if self.score.rankall:
                return self.market[factor_name]
            else:
                if self.pool.inexclude=='exclude':
                    # 排除掉的置为np.nan
                    return self.market[factor_name]*(self.market[self.pool.inexclude].astype(int).\
                                                replace(to_replace={1:np.nan, 0:1}))
                else:
                    return self.market[factor_name]*(self.market[self.pool.inexclude].astype(int).\
                                                replace(to_replace={0:np.nan}))
        # 获取打分
        for factor in self.score.exp:
            self.market[factor[0]+'_score'] = process_factor(factor[0]).groupby('date').\
                rank(ascending=factor[1])*factor[2]
        basescore = [i[0]+'_score' for i in self.score.exp]
        self.market['score'] = self.market[basescore].sum(axis=1)
        #print(time.time()-time0)
        #time0 = time.time()
        # 回测
        strat0 = FB.strat.MetaStrat(self.market, self.pool.inexclude, 'score',\
                                     self.hold_num, self.price)
        strat0.run()
        #print(time.time()-time0)
        #time0 = time.time()
        # 后处理
        self.post = FB.post.StratPost(strat0, self.market, comm=self.comm, fast=True)
        #print(time.time()-time0)
        #time0 = time.time()
        return self.post
