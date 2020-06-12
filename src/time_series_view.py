from TestEval import ContinuousTestEval
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# WORK IN PROGRESS - currently does not work


# lots of code to create time list for testing
dates = [pd.to_datetime('2019-06-01') + dt.timedelta(days = i) for i in range(10)]
multiplier = lambda x: [x for i in range(10000)]

date_ll = []

for i in dates:
    date_ll.append(multiplier(i))

dummy_dates = []

for i in date_ll:
    for j in i:
        dummy_dates.append(j)
# end of date stuff

CONTROL = pd.DataFrame({'data':np.random.normal(0, 2, len(dummy_dates)),
                        'date':dummy_dates}
                       ).sort_values(by = 'date')
TEST = pd.DataFrame({'data':np.random.random(len(dummy_dates)) * 300,
                    'date':dummy_dates}
                    ).sort_values(by = 'date')

QUANTILES = np.arange(.1, 1, .2)

# TODO create pval df function
# TODO create pval chart function
# TODO create metric avg performance function

class TimeSeries():
    def __init__(self, control, test):
        self.control = control
        self.test = test

        assert isinstance(self.control, pd.core.frame.DataFrame) and isinstance(self.test, pd.core.frame.DataFrame), \
                        'control and test need to be pandas data frames'

        assert self.control.shape[1] == 2 and self.control.shape[1] == 2, \
                        'control and test data needs to have two columns'

    def __repr__(self):
        return 'Class to examine a/b testing over time'


    def pvalue_df(self):

        temp_ctrl = self.control #.groupby('date').mean().reset_index()
        temp_test = self.test #.groupby('date').mean().reset_index()

        date_list = []
        pval_list = []

        for i in self.control['date'].unique():
            date_list.append(i)
            ctrl = temp_ctrl.loc[temp_ctrl['date'] <= i]
            test = temp_test.loc[temp_test['date'] <= i]
            pval_calc = ContinuousTestEval(ctrl['data'],
                                           test['data'])
            pval_list.append(pval_calc.continuous_pval(n = 400))

        output_df = pd.DataFrame({'dates':date_list,
                                  'pvalues':pval_list}
                                 ).sort_values(by = 'dates')

        return output_df


    def quantiled_treatment_df(self):
        return None


    def pvalue_viz(self):

        pval_df = self.pvalue_df()

        fig, ax = plt.subplots(1, 1, figsize = (10, 5))

        ax.plot(pval_df['dates'], pval_df['pvalues'], color = 'b')
        ax.axhline(y = 0.05, ls = '--', lw = '1', color = 'k')

        plt.show()


    def metric_viz(self):

        temp_ctrl = self.control.groupby('date').mean().reset_index()
        temp_test = self.test.groupby('date').mean().reset_index()

        fig, ax = plt.subplots(1, 1, figsize = (10, 6))

        ax.plot(temp_ctrl['date'], temp_ctrl['data'], color = 'b', label = 'Control')
        ax.plot(temp_test['date'], temp_test['data'], color = 'r', label = 'Test')
        ax.legend()

        plt.show()


    def quantiled_treatment_viz(self):
        return None

a = TimeSeries(CONTROL, TEST)

a.pvalue_viz()
# a.metric_viz()