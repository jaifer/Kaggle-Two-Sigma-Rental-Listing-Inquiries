import numpy as np
import xgboost as xgb

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
        blend_train = np.zeros((self.train_y.shape[0], 1))
        blend_test = np.zeros((self.test.shape[0], len(self.skf)))
        for i, (train_i, val_i) in enumerate(self.skf):
            val_train_x = self.train_x[train_i]
            val_train_y = self.train_y[train_i]
            val_test_x = self.train_x[val_i]
            val_test_y = self.train_y[val_i]

            train_d = xgb.DMatrix(val_train_x, label=val_train_y)
            test_d = xgb.DMatrix(val_test_x, label=val_test_y)
            watchlist = [ (train_d,'train'), (test_d, 'test') ]

            model = xgb.train(self.params, train_d, evals=watchlist, num_boost_round=100000, early_stopping_rounds=25, verbose_eval=False)
            avg_iteration += model.best_iteration

            val_pred_y = model.predict(test_d)
            pred_y = model.predict(xgb.DMatrix(self.test))

            blend_train[val_i, 0] = val_pred_y
            blend_test[:, i] = pred_y
            score = self.eval_func(val_test_y, val_pred_y)

        score = self.eval_func(self.train_y, blend_train)

        avg_iteration = int(avg_iteration * (1.0 + 1.0/len(self.skf)))
        train_d = xgb.DMatrix(self.train_x, label=self.train_y)
        test_d = xgb.DMatrix(self.test)
        model = xgb.train(self.params, train_d, num_boost_round=avg_iteration)
        pred_y = model.predict(test_d)

        return pred_y, blend_train.ravel(), blend_test.mean(1), score
