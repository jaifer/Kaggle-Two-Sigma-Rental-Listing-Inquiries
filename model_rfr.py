import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Model():

    def __init__(self, train_x, train_y, test, skf, eval_func, params):
        self.train_x = train_x
        self.train_y = train_y
        self.test = test
        self.skf = skf
        self.eval_func = eval_func
        self.p_n_estimators = params.get('n_estimators', 100)
        self.p_min_samples_split = params.get('min_samples_split', 2)
        self.p_min_samples_leaf = params.get('min_samples_leaf', 4)
        self.p_max_features = params.get('max_features', 'auto')
        self.p_max_depth = params.get('max_depth', None)
        self.seed = params.get('seed', 1234)

    def Predicting(self):
        score = 0
        blend_train = np.zeros((self.train_y.shape[0], 1))
        blend_test = np.zeros((self.test.shape[0], len(self.skf)))
        for i, (train_i, val_i) in enumerate(self.skf):
            val_train_x = self.train_x[train_i]
            val_train_y = self.train_y[train_i]
            val_test_x = self.train_x[val_i]
            val_test_y = self.train_y[val_i]

            model = RandomForestRegressor(n_estimators=self.p_n_estimators, max_features=self.p_max_features, min_samples_split=self.p_min_samples_split, min_samples_leaf=self.p_min_samples_leaf, max_depth=self.p_max_depth, random_state=self.seed, n_jobs=-1)
            model.fit(val_train_x, val_train_y)
            val_pred_y = model.predict(val_test_x)
            pred_y = model.predict(self.test)

            blend_train[val_i, 0] = val_pred_y
            blend_test[:, i] = pred_y
            score = self.eval_func(val_test_y, val_pred_y)

        score = self.eval_func(self.train_y, blend_train)

        model = RandomForestRegressor(n_estimators=self.p_n_estimators, max_features=self.p_max_features, min_samples_split=self.p_min_samples_split, min_samples_leaf=self.p_min_samples_leaf, max_depth=self.p_max_depth, random_state=self.seed, n_jobs=-1)
        model.fit(self.train_x, self.train_y)
        pred_y = model.predict(self.test)

        return pred_y, blend_train.ravel(), blend_test.mean(1), score

