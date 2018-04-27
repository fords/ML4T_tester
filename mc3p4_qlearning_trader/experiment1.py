
# Zeyar Win(zwin3)

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode
import StrategyLearner as sl
import ManualStrategy as ms

def author():
    return('zwin3')

if __name__ == "__main__":
    print('  Experiment 1 ')
    sym = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sd2 = dt.datetime(2010,1,1)
    ed2 = dt.datetime(2011,12,31)
    sv = 100000
    learner = sl.StrategyLearner(verbose=False, impact=0.00)
    learner.addEvidence(sym, sd, ed, 100000)
    strategy = learner.testPolicy(sym, sd, ed)
    manual = ms.testPolicy(sym, sd, ed)
    benchmark = pd.DataFrame(index=strategy.index)
    benchmark[sym] = 0
    benchmark.iloc[0,0] = 1000
    values = marketsimcode.compute_portvals(strategy)
    values_bench = marketsimcode.compute_portvals(benchmark)
    values_manual = marketsimcode.compute_portvals(manual)
    values /= sv
    values_bench /= sv
    values_manual /= sv
    fig = plt.figure(figsize=(10,5), dpi=80)
    plt.plot(values, color='b', label='Strategy')
    plt.plot(values_bench, color='r', linestyle=':', linewidth=2, label='Benchmark')
    plt.plot(values_manual, color='y', label='Manual')
    plt.xlabel('Dates', fontsize=12)
    plt.ylabel('Portfolio Data', fontsize=12)
    fig.suptitle('In Sample Data:  Benchmark Vs. Manual Vs. Strategy', fontsize=20)
    plt.show()
    columns = len(learner.xTrain.columns)
    result_df = pd.concat([values, values_bench, values_manual],axis=1)
    result_df.rename(columns={0 : 'Strategy_Learner', 1 : 'Benchmark', 2 : 'Manual_Strategy'}, inplace=True)
    print('Performance result : ')
    print(result_df)
