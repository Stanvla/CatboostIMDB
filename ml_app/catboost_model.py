# %%
import catboost
import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import yaml


class GBTrees:
    sigmoid = lambda y, x: 1 / (1 + np.exp(-x))

    def __init__(self,
                 iterations,
                 learning_rate,
                 eval_metric,
                 leaf_estimation_method,
                 l2_leaf_reg,
                 text_processing_params_json,
                 random_seed,
                 model_fn,
                 train_pool=None,
                 val_pool=None):
        with open(text_processing_params_json, 'r') as f:
            text_processing_params = json.load(f)

        self.model = catboost.CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            eval_metric=eval_metric,
            leaf_estimation_method=leaf_estimation_method,
            l2_leaf_reg=l2_leaf_reg,
            text_processing=text_processing_params,
            use_best_model=True,
            random_seed=random_seed,
        )

        if not self.load_model(model_fn):
            self._check_data(train_pool, 'train')
            self._check_data(val_pool, 'validation')

            self.model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=200,
            )
            self.decision_boundary = self._select_decision_boundary(val_pool)
            self.save_model(model_fn)

    def _select_decision_boundary(self, pool):
        logits = self.model.predict(
            data=pool,
            prediction_type='RawFormulaVal',
        )
        probs = self.sigmoid(logits)
        labels = pool.get_label()
        result = dict(
            decision_boundary=0.5,
            acc=np.sum((probs > 0.5) == labels)/len(probs)
        )
        for p in probs:
            predictions = probs > p
            acc = np.sum(predictions == labels) / len(predictions)
            if acc > result['acc']:
                result['acc'] = acc
                result['decision_boundary'] = p
        return result['decision_boundary']

    def _check_data(self, data, type):
        if data is None:
            raise ValueError(f'The model is not trained and {type} pool is None')

    def load_model(self, model_fn):
        if os.path.isfile(model_fn):
            self.model.load_model(model_fn)
            decision_boundary_fn = model_fn.replace('.bin', '_db.pkl')
            with open(decision_boundary_fn, 'rb') as f:
                self.decision_boundary = pickle.load(f)
            return True
        return False

    def save_model(self, model_fn):
        self.model.save_model(model_fn)
        decision_boundary_fn = model_fn.replace('.bin', '_db.pkl')
        with open(decision_boundary_fn, 'wb') as f:
            pickle.dump(self.decision_boundary, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_item(self, item):
        x = [item['review']]
        prob = self.model.predict(x, prediction_type='Probability')[1]
        if prob > self.decision_boundary:
            return 'Positive'
        return 'Negative'

    def eval_model(self, eval_set):
        logits = self.model.predict(
            data=eval_set,
            prediction_type='RawFormulaVal'
        )
        probs = self.sigmoid(logits)
        return np.sum((probs > self.decision_boundary) == eval_set.get_label()) / len(probs)


def get_train_val_test_pools(csv_file, val_frac, test_frac, seed):
    df = pd.read_csv(csv_file)
    # encode sentiment
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    df.drop(['sentiment'], axis=1, inplace=True)

    X, y = df.drop(columns=['label']), df.label
    test_val_frac = val_frac + test_frac

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=1 - test_val_frac, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=val_frac/test_val_frac, random_state=seed)
    train_pool = catboost.Pool(data=X_train, label=y_train, text_features=['review'])
    val_pool = catboost.Pool(data=X_val, label=y_val, text_features=['review'])
    test_pool = catboost.Pool(data=X_test, label=y_test, text_features=['review'])

    return train_pool, val_pool, test_pool


# %%
if __name__ == '__main__':
    params_fn = 'configs/catboost_config.yml'
    with open(params_fn, 'r') as f:
        params = yaml.safe_load(f)

    train_pool, val_pool, test_pool = get_train_val_test_pools(**params['data_params'])
    catboost_model = GBTrees(**params['catboost_params'], train_pool=train_pool, val_pool=val_pool)
    print(catboost_model.eval_model(val_pool))
    print(catboost_model.eval_model(test_pool))