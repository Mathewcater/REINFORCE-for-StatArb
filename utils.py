import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch as T
from main_training import *
from Simulator import *

mblue = (0.098,0.18,0.357)
mred = (0.902,0.4157,0.0196)
mgreen = (0.,0.455,0.247)
mpurple = (0.5804,0.2157,0.9412)
mgray = (0.5012,0.5012,0.5012)
myellow = (0.8,0.8,0)
mwhite = (1.,1.,1.)
cmap = LinearSegmentedColormap.from_list('beamer_cmap', [mred, mwhite, mblue])
colors = [mblue, mred, mgreen, myellow, mpurple, mgray]