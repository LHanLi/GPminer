
############################################# 参数设置 ###################################################
max_g = 10 # 进化代数
population_size = 10 # 种群规模
evolution_ratio = 0.5 # 每一代选择比例
prob_dict = {}
#prob_dict = {'mutation_exp_dw':0.3, 'mutation_exp_replace':0.3, 'mutation_exp_and':0.3,\
#             'cross_exp_exchange':0.1}  # 各种变异、联会发生概率
select_alg = 'cut'  # 选择算法 wheel tourment
# 策略超参数
start = '2022-8-1'
end = '2024-10-10'
price = 'price' # 结算价格
# 所有策略共用一个cond
pool0 = ind.Pool("left_years<1|close<100|close>135|is_call=公告实施强赎|"\
    "is_call=公告提示强赎|is_call=已满足强赎条件|is_call=公告到期赎回|list_days<4")

# 基础因子
date_factors = ['dur_days', 'last_days', 'a_dur_days']
trade_factors = ['amount', 'turnover', 'a_amount', 'a_turnover']
size_factors = ['Cap', 'balance', 'a_Cap', 'a_totalCap', 'CapRatio']
bond_factors = ['ytm', 'pure_bond', 'return_money', 'hold_return']
option_factors = ['premium', 'DoubleCheap', 'volatility', 'PcvB', 'option_value', 'mod_option_value']
basket = date_factors+trade_factors+size_factors+bond_factors+option_factors

basket = ['conv_prem', 'theory_conv_prem', 'mod_conv_prem', 'close', 'dblow', \
               'pure_value', 'bond_prem', 'remain_size', 'remain_cap',\
                'conv_value', 'option_value', 'theory_value', 'theory_bias', 'pct_chg',\
                'turnover', 'cap_mv_rate', 'list_days', 'ytm', 'pct_chg_5',\
                'pct_chg_5_stk', 'volatility', 'volatility_stk', 'close_stk', \
                'pct_chg_stk', 'amount_stk', 'vol_stk', 'total_mv', 'circ_mv', \
                    'pb', 'pe_ttm', 'ps_ttm', 'debt_to_assets', 'dv_ratio']

############################################ 数据 ##############################################
market


############################################ 初始化 #############################################
gen0 = gen.Gen(basket)
seeds = gen0.get_seeds()
ino.save_pkl(seeds, 'seeds', 'data')


######################################### 测试最有并行核数 ######################################
np = 100
eval0 = eval.Eval(market, pool0)
def test_ncore(n_core, popu0):
    t0 = time.time()
    # 计算适应度
    def single(p):
        result = pd.DataFrame()
        post0 = eval0.eval_score(p)
        result.loc[p, 'return'] = post0.return_annual*100
        result.loc[p, 'sharpe'] = post0.sharpe
        result.loc[p, 'drawdown'] = post0.drawdown.max()*100
        return result
    Parallel(n_jobs=n_core)(delayed(single)(p) \
                                    for p in list(popu0.codes)) 
    return time.time()-t0
popu0 = popu.Population()
popu0.reset(set(list(ino.read_pkl('seeds', 'data'))[:np]))
for n_core in range(20, 2, -2):
    print(n_core, '核并行耗时', test_ncore(n_core, popu0), '秒')

#################################### Run #####################################################

def workflow():
    init_population_name = 'undefine'
    fitness = pd.DataFrame()
    eval0 = eval.Eval(market, pool0)
    for g in range(max_g):
        t0 = time.time()
        if g==0:
            # 生成初代种群
            popu0 = popu.Population()
            popu0.add(ino.read_pkl('seeds', 'data'))
            print('从%s个p中选择%s个p作为初始种群'%(len(popu0.codes),\
                                    int(population_size/evolution_ratio)))
            popu0 = popu0.subset(int(population_size/evolution_ratio))
            popu0.get_name(2)
            init_population_name = datetime.datetime.now().date().strftime("%Y%m%d")+popu0.name
            os.mkdir(init_population_name)
        # 计算适应度
        def single(p):
            result = pd.DataFrame()
            post0 = eval0.eval_score(p)
            result.loc[p, 'return'] = post0.return_annual*100
            result.loc[p, 'sharpe'] = post0.sharpe
            result.loc[p, 'drawdown'] = post0.drawdown.max()*100
            return result
        if g!=0:
            # 上一代已有数据无需计算
            fitness = fitness.loc[list(popu0.codes&set(fitness.index))]
        print('本代', len(popu0.codes), '个策略，其中', len(fitness), '个策略已有计算结果')
        fitness_list = Parallel(n_jobs=n_core)(delayed(single)(p) \
                                        for p in list(popu0.codes-set(fitness.index)))
        fitness = pd.concat([fitness, pd.concat(fitness_list)])
        #fitness.to_feather(init_population_name+'/'+'fitness%s.feather'%g)
        fitness = fitness.sort_values('sharpe', ascending=False)
        fitness.to_csv(init_population_name+'/'+'fitness%s.csv'%g)
        # 选择
        popu0.codes = set(fitness[:population_size].index) # 截断选择 # 锦标赛选择# 轮盘赌选择
        print('第%s轮进化完成，耗时%.1lfs，最大夏普%.2lf'%(g, time.time()-t0, fitness.iloc[0]['sharpe']))
        if g==(max_g-1):
            break
        # 种群繁殖
        gen0 = gen.Gen(basket)
        gen0.popu = popu0
        gen0.multiply(1/evolution_ratio)

workflow()

