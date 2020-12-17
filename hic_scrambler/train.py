# Script for training and saving SV prediction model
# cmdoret, 20190515
import model as ml
from os.path import join

PROFILE = 'Debug'
DATA_PATH = join('data/input/training', PROFILE)
MODEL_PATH = join('data/models', PROFILE)

x_data, y_data = ml.load_data(DATA_PATH)
sv_model = ml.create_model()
sv_model.fit(x_data, y_data, epochs=15)
ml.save_model(sv_model, MODEL_PATH)
