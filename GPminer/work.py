import pandas as pd
import numpy as np
from random import sample
from joblib import Parallel, delayed
import GPminer as GPm
import FreeBack as FB
import time, datetime, os


class Miner():
    # 矿工初始化需输入策略超参数，share表示挖掘出一组策略的共享部分，
    # 如果share是pool则挖掘score，反之亦然，如果是None则挖掘SP
    def __init__(self, market, benchmark=None, share=None, pool_basket=None, score_basket=None, p0=None,\
                  hold_num=5, comm=10/1e4, price='close'):
        self.market = market
        self.benchmark = benchmark
        self.share = share
        if pool_basket!=None:
            self.pool_basket = pool_basket
        else:
            self.pool_basket = list(market.columns)
        if score_basket!=None:
            self.score_basket = score_basket
        else:
            self.score_basket = list(market.columns)
        self.p0 = p0
        self.hold_num = hold_num
        self.comm = comm
        self.price = price
    def prepare(self, fitness='sharpe',\
                 population_size=10, evolution_ratio=0.2, tolerance_g=3, max_g=10,\
                  prob_dict={}, select_alg='cut', n_core=4):
        self.fitness = fitness
        self.population_size = population_size
        self.evolution_ratio = evolution_ratio
        self.tolerance_g = tolerance_g
        self.max_g = max_g
        self.prob_dict = prob_dict
        self.select_alg = select_alg
        self.n_core = n_core
        # 生成初代种群需要
        self.gen0 = GPm.gen.Gen(score_basket=self.score_basket,\
                            market=self.market, indtype='Score')
        if self.p0!=None:
            self.gen0.popu.add(self.p0.code)
            while len(self.gen0.popu.codes)<int(self.population_size/self.evolution_ratio):
                self.gen0.multiply()
        else:
            self.seeds = list(self.gen0.get_seeds())
    def run(self):
        workfile = datetime.datetime.now().strftime("%m%d%H%M_%S_%f")+\
                            '_%s'%np.random.rand()
        t0 = time.time()
        if self.p0!=None:
            init_seeds = self.gen0.popu.subset(int(self.population_size/self.evolution_ratio)\
                                               -1).codes|{self.p0.code}
        else:
            init_seeds = sample(self.seeds, int(self.population_size/self.evolution_ratio))
        GPm.ino.log('生成%s个p作为初始种群'%len(init_seeds))
        GPm.ino.log('=====此初始种群进化开始=====')
        os.mkdir(workfile)
        fitness_all = pd.DataFrame()
        fitness_df = pd.DataFrame()
        # 后续进化在popu0上操作 
        popu0 = GPm.popu.Population(type=GPm.ind.Score)
        eval0 = GPm.eval.Eval(self.market, pool=self.share)
        eval0.eval_pool()
        for ind in init_seeds:
            popu0.add(ind) 
        gen0 = GPm.gen.Gen(score_basket=self.score_basket, market=self.market,\
                            indtype='Score', popu0=popu0)
        max_fitness = -99999
        max_loc = 0
        for g in range(self.max_g):
            GPm.ino.log('第%s代'%(g))
            # 计算适应度
            def single(p):
                result = pd.DataFrame()
                eval0.eval_score(p)
                strat0 = eval0.backtest(self.hold_num, self.price)
                post0 = FB.post.StratPost(strat0, eval0.market, benchmark=self.benchmark,\
                                            comm=self.comm, show=False)
                result.loc[p, 'return_total'] = post0.return_total
                result.loc[p, 'return_annual'] = post0.return_annual
                result.loc[p, 'sigma'] = -post0.sigma
                result.loc[p, 'sharpe'] = post0.sharpe
                result.loc[p, 'drawdown'] = -max(post0.drawdown)
                result.loc[p, 'excess_annual'] = post0.excess_return_annual
                result.loc[p, 'excess_sigma'] = -post0.excess_sigma
                result.loc[p, 'excess_sharpe'] = post0.excess_sharpe
                result.loc[p, 'excess_drawdown'] = -max(post0.excess_drawdown)
                result.loc[p, 'beta'] = post0.beta
                result.loc[p, 'alpha'] = post0.alpha*250*100
                return result
            if g!=0:
                # 之前已经计算过的无需计算
                fitness_df = fitness_all.loc[list(popu0.codes&set(fitness_all.index))]
            GPm.ino.log('本代%d个策略，其中%d个策略已有计算结果'%(len(popu0.codes), len(fitness_df)))
            if len(popu0.codes)!=len(fitness_df):
                if self.n_core!=1:   # 并行
                    fitness_list = Parallel(n_jobs=self.n_core)(delayed(single)(p) \
                                            for p in list(popu0.codes-set(fitness_df.index)))
                else:    # 串行
                    fitness_list = []
                    for p in list(popu0.codes - set(fitness_df.index)):
                        fitness_list.append(single(p))
                fitness_all = pd.concat([fitness_all, pd.concat(fitness_list)]).drop_duplicates()
                fitness_df = pd.concat([fitness_df, pd.concat(fitness_list)])
            GPm.ino.log('第%s轮进化适应度计算完成'%g)
            fitness_df = fitness_df.sort_values(self.fitness, ascending=False)
            if fitness_df.iloc[0][self.fitness]>max_fitness:
                max_fitness = fitness_df.iloc[0][self.fitness]
                max_loc = g
            fitness_df.to_csv(workfile+'/fitness%s.csv'%g)
            # 选择
            if self.select_alg=='cut':
                popu0.reset(set(fitness_df[:self.population_size].index)) # 截断选择
            # 锦标赛，不放回
            elif self.select_alg=='tournament':
                select = set()
                while len(select)<self.population_size:
                    one = set(fitness_df.loc[sample(list(set(fitness_df.index)-select), int(len(fitness_df)/10))]\
                                .sort_values(by=self.fitness, ascending=False).index[:1])
                    select = select|one
                popu0.reset(select)
            GPm.ino.log('第%s轮进化完成，最大%s:%.2lf'%(g, self.fitness, fitness_df.iloc[0][self.fitness]))
            if ((g-max_loc)>=self.tolerance_g)|(g>=(self.max_g-1)):
                cost = time.time()-t0
                GPm.ino.log('=====此初始种群进化完成=====共计算%d个策略，总耗时%.1lfs，单策略耗时%.2lfs'%(\
                    len(fitness_all), cost, cost/len(fitness_all)))
                fitness_df.loc[list(popu0.codes)].sort_values(by=self.fitness, ascending=False).\
                    to_csv(workfile+'/fitness%s.csv'%(g+1))
                # 重命名结果
                os.rename(workfile, 'result-' + workfile+'-'+popu0.get_name())
                break
            # 种群繁殖
            gen0.multiply(1/self.evolution_ratio)
            GPm.ino.log('交叉变异生成第%s代种群'%(g+1))


            