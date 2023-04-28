import pandas as pd
from fbprophet import Prophet

df = pd.read_csv("timeseries0.csv")
m = Prophet()
m.fit(df)

df = pd.read_csv("timeseries1.csv")
m2 = Prophet()
def stan_init2():
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res
m2.fit(df, init=stan_init2)
