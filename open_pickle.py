import pickle

from pathlib import Path


pickle_name = Path('complete_models/test/winner7.pkl')
pickle_in = open(pickle_name,"rb")
model = pickle.load(pickle_in)

print(model)
