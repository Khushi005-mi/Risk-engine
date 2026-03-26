import numpy as np

from engine.modeling.gbdt_model import GBDTModel


def test_model_training():

    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    model = GBDTModel()

    model.train(X, y)

    assert model.model is not None


def test_prediction_shape():

    X = np.random.rand(10, 5)

    model = GBDTModel()

    model.model = model._init_model()

    preds = model.predict_proba(X)

    assert preds.shape[0] == 10