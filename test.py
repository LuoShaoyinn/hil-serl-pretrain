#! python3

import code
import pickle

with open("resnet10_params.pkl", "rb") as file:
    x = pickle.load(file)

code.interact(local=locals())
