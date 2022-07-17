from flask import Flask, request
import yaml
from catboost_model import GBTrees, get_train_val_test_pools
import os

app = Flask(__name__)
# print(os.getcwd())
# print(os.listdir())

params_fn = 'ml_app/configs/catboost_config.yml'
with open(params_fn, 'r') as f:
    params = yaml.safe_load(f)

catboost_model = GBTrees(**params['catboost_params'])
_, _, test_pool = get_train_val_test_pools(**params['data_params'])


@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    if 'review' not in features:
        return dict(result='Request should have "review" field.')

    result = catboost_model.predict_item(features)
    return dict(
        review=features['review'],
        sentiment=result
    )


@app.route('/eval', methods=['GET'])
def eval():
    score = catboost_model.eval_model(test_pool)
    return dict(
        resutl=f'Accuracy on the test set is {score * 100:.2f}%.'
    )


if __name__ == '__main__':
    app.run()
