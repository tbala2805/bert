# datarobot_hook.py
import os
from setfit import SetFitModel

model = None

def load_model(code_dir, **kwargs):
    global model
    model_path = os.path.join(code_dir, "setfit_model")
    model = SetFitModel.from_pretrained(model_path)

def score(data, **kwargs):
    texts = data["text"].tolist()
    predictions = model.predict(texts)
    return predictions
