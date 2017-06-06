import numpy as np
import lightgbm as lgbm

class Model():

    def __init__(self, train_x, train_y, test, skf, eval_func, params):
        self.train_x = train_x
        self.train_y = train_y
        self.test = test
        self.skf = skf
        self.eval_func = eval_func
        self.params = params

    def Predicting(self):
        score = 0
        avg_iteration = 0
        blend_train = np.zeros((self.train_y.shape[0], self.params['num_class']))
        blend_test = np.zeros((self.test.shape[0], len(self.skf), self.params['num_class']))
        for i, (train_i, val_i) in enumerate(self.skf):
            val_train_x = self.train_x[train_i]
            val_train_y = self.train_y[train_i]
            val_test_x = self.train_x[val_i]
            val_test_y = self.train_y[val_i]

            train_d = lgbm.Dataset(val_train_x, val_train_y, silent=True)
            test_d = lgbm.Dataset(val_test_x, val_test_y, silent=True)
            watchlist = [ (val_test_x, val_test_y) ]

            model = lgbm.train(self.params, train_d, valid_sets=[test_d], valid_names=['test'], num_boost_round=100000, early_stopping_rounds=25, verbose_eval=False)
            
            avg_iteration += model.best_iteration
            
            val_pred_y = model.predict(val_test_x, num_iteration=model.best_iteration)
            pred_y = model.predict(self.test, num_iteration=model.best_iteration)

            blend_train[val_i, :] = val_pred_y
            blend_test[:, i, :] = pred_y
            score = self.eval_func(val_test_y, val_pred_y)

        score = self.eval_func(self.train_y, blend_train)

        avg_iteration = int(avg_iteration * (1.0 + 1.0/len(self.skf)))
        train_d = lgbm.Dataset(self.train_x, self.train_y)
        model = lgbm.train(self.params, train_d, num_boost_round=avg_iteration)
        pred_y = model.predict(self.test)

        return pred_y, blend_train, blend_test.mean(1), score

