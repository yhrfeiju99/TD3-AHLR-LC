import numpy as np
import sys
import random
from tqdm import tqdm
from joblib import dump
from utils import *
from lgb import *
from tf2rl import *
from config import *
from simulation_env import *


TTC_threshold=3.001
MaxCrsLoc=3.49
MaxChangeLaneTime=225

timeWindow=3 

env = Env(TTC_threshold,MaxCrsLoc,MaxChangeLaneTime,timeWindow)
s_dim = env.n_features
a_dim = env.n_actions

trainNum = train.shape[0]
testNum = test.shape[0]
print('Number of training samples:', trainNum)
print('Number of validate samples:', testNum)

n_run = 2000
rolling_window = 5
result = []

for run in [base_name]:
    td3_agent = TD3(a_dim, s_dim, a_bound)
    max_rolling_score = np.float64('-inf')
    max_score = np.float64('-inf')
    nos_var = 0.1
    collision_train = 0
    episode_score = np.zeros(total_episode)
    rolling_score = np.zeros(total_episode)
    cum_collision_num = np.zeros(total_episode)
    episode_score_info = np.zeros((total_episode, 15))
    
    for i in tqdm(range(total_episode)):
        car_fol_id = random.randint(0, trainNum - 1)
        data = train[car_fol_id, 0]
        s = env.reset(data)
        s = (s - np.tile(s_mean, timeWindow)) / np.tile(s_std + 1, timeWindow)
        
        # Episode initialization
        score = 0
        score_info = np.zeros(15)
        action_lst = []
        start_learning = False
        
        while True:
            a = td3_agent.choose_action(s, add_noise=True)
            a1, a2 = a[0], a[1]
            action_lst.append(np.array([a1, a2]))
            
            s_, r, done, r_info = env.step([a1, a2])
            s_ = (s_ - np.tile(s_mean, timeWindow)) / np.tile(s_std + 1, timeWindow)
            
            td3_agent.store_transition(s, a, r, s_, done)
            
            if td3_agent.pointer > MEMORY_CAPACITY * 1:
                start_learning = True
                nos_var = max(0.0001, nos_var * 0.98)
                td3_agent.learn()
            
            s = s_
            score += r
            score_info += r_info
            
            # Terminate episode if done
            if done:
                if env.isCollision == 1:
                    collision_train += 1
                break
        
        episode_score[i] = score
        episode_score_info[i] = score_info
        rolling_score[i] = np.mean(episode_score[max(0, i - rolling_window + 1):i + 1])
        cum_collision_num[i] = collision_train
        
        if score > max_score:
            max_score = score
        if rolling_score[i] > max_rolling_score:
            max_rolling_score = rolling_score[i]
            td3_agent.save_model(f'{run}')
        
        sys.stdout.write(
            f'\r Run {run}, Start learning: {start_learning}, Episode {i+1}, Action mean {np.mean(action_lst, axis=0)}, '
            f'Score: {score:.5f}, Rolling score: {rolling_score[i]:.5f}, Max score: {max_score:.5f}, '
            f'Max rolling score: {max_rolling_score:.5f}, collisions: {collision_train}   '
        )
        sys.stdout.flush()
    

    result.append([episode_score, rolling_score, cum_collision_num, episode_score_info])

dump(result, f'result_{run}.joblib')

