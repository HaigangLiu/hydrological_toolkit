
import pymc3 as pm
# from data import sample_data

# rainfall_data = sample_data.load_monthly_rain()
# rainfall_data_jan = rainfall_data[rainfall_data.MONTH == 12]
# print(rainfall_data_jan)

import numpy as np
import pandas as pd
from model.gaussian_process import GP


if __name__ =='__main__':




    n = 10  # The number of data points
    X = np.linspace(0, 10, n)[:, None]  # The inputs to the GP, they must be arranged as a column vector

    l_true = 1.0
    e_true = 3
    cov_func = e_true ** 2 * pm.gp.cov.Matern52(1, l_true)

    mean_func = pm.gp.mean.Zero()
    f_true = np.random.multivariate_normal(mean_func(X).eval(),
                                           cov_func(X).eval() + 1e-8 * np.eye(n), 1).flatten()

    n_true = 3.0
    y = f_true + np.random.normal(loc=0, scale=n_true, size=n)

    # import pickle
    # with open('test_cache.pkl', 'rb') as reader:
    #     trained_model = pickle.load(reader)
    # t = trained_model.trace
    # print(y)
    # print(t['f'].mean(axis=0))
    #
    # exit()

    gaussian_process_model = GP()
    df = pd.DataFrame([X.flatten(), y]).T
    df.columns = ['x', 'y']
    gaussian_process_model.fit(df=df,
                               locations_cols=['x'],
                               y_col=['y'],
                               draws=5000)
    predicted = gaussian_process_model.predict()
    result = gaussian_process_model.summarize()

    print(result)
    print(predicted)
    print(f_true)
    # import pickle
    # with open('test_cache.pkl', 'wb') as writer:
    #     pickle.dump(gaussian_process_model, writer, protocol=pickle.HIGHEST_PROTOCOL)

    # exit()
    # print(gaussian_process_model.trace['y'])
    # prediction = gaussian_process_model.predict(df=df, location_cols=['x'])
    # print(prediction)
    # result = gaussian_process_model.summarize()
    # print(result)