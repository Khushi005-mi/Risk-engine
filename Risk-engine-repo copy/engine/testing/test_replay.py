import numpy as np

from engine.modeling.inference_model import InferenceModel


def test_prediction_reproducibility():

    model = InferenceModel()

    X = np.random.rand(10, 5)

    p1 = model.predict(X)
    p2 = model.predict(X)

    assert (p1 == p2).all()