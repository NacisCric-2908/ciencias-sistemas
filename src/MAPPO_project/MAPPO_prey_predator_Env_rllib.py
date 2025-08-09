import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ray.rllib.env import PettingZooEnv
from pettingzoo.utils import parallel_to_aec
from Environment.prey_predator_Env import PreyPredatorEnv


""" This function creates the environment for the MAPPO algorithm. """
def env_creator(config=None):
    return PettingZooEnv(parallel_to_aec(PreyPredatorEnv()))
