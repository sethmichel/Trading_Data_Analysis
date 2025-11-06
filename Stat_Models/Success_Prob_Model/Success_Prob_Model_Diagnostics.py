import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pygam import LogisticGAM, s, te
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt


version = None
success_prob_dir = "Holder_Strat/Parameter_Tuning/Success_Prob_Model"
def Set_Version(passed_version):
    global version
    version = passed_version