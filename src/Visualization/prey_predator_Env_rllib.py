# prey_predator_env_rllib.py
from prey_predator_Env import PreyPredatorEnv  

"""
This file wrapped the PettingZoo env to use rllib functions. 
"""

def raw_env():
    return PreyPredatorEnv()

def env(render_mode):
    #print("✅ env() fue llamado")
    return PreyPredatorEnv(render_mode)