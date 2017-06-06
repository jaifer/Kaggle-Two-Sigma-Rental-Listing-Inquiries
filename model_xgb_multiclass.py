import numpy as np
import xgboost as xgb
import operator

class Model():

    def __init__(self, train_x, train_y, test, skf, eval_func, params, num_class=3):
        self.train_x = train_x
        self.train_y = train_y
        self.test = test
        self.skf = skf
        self.eval_func = eval_func
        self.params = params
        self.num_class = num_class

    def Predicting(self):
        score = 0
        avg_iteration = 0
        blend_train = np.zeros((self.train_y.shape[0], self.num_class))
        blend_test = np.zeros((self.test.shape[0], len(self.skf), self.num_class))
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

            blend_train[val_i, :] = val_pred_y
            blend_test[:, i, :] = pred_y
            score = self.eval_func(val_test_y, val_pred_y)

        score = self.eval_func(self.train_y, blend_train)

        avg_iteration = int(avg_iteration * (1.0 + 1.0/len(self.skf)))
        train_d = xgb.DMatrix(self.train_x, label=self.train_y)
        test_d = xgb.DMatrix(self.test)
        model = xgb.train(self.params, train_d, num_boost_round=avg_iteration)
        pred_y = model.predict(test_d)

#        feat_imp = model.booster().get_fscore()
#        feat_imp = sorted(feat_imp.items(), key=operator.itemgetter(1))
#        print(feat_imp)
        
        return pred_y, blend_train, blend_test.mean(1), score

