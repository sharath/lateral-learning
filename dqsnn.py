import os, sys
import random
import pickle
import itertools

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

from bindsnet.network import Network
from bindsnet.network.nodes import RealInput, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.environment import GymEnvironment 
from bindsnet.learning import WeightDependentPostPre as wd_post_pre
from bindsnet.learning import PostPre as post_pre
from bindsnet.learning import MSTDP as mstdp
from gym import wrappers
from time import time as T
from argparse import ArgumentParser
from collections import deque, namedtuple

experiment_dir = f'{int(T())}'
    

# definition of DQANN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def transfer(ann_path, dt=1.0, runtime=500, scale1=6.452, scale2=71.155, stdp=0, nu_pre=1e-8, nu_post=1e-6, rand_init=0, device_id=0):
    if not torch.cuda.is_available():
        DQANN = torch.load(ann_path, map_location='cpu')
    else:
        DQANN = torch.load(ann_path).cuda(device_id)
    
    DQSNN = Network(dt=dt)
    
    # Create Layers.
    inpt = RealInput(n=6400, traces=False)
    exc = LIFNodes(n=1000, refrac=0, traces=True, thresh=-52.0, rest=-65.0, decay=1e-2)
    readout = LIFNodes(n=4, refrac=0, traces=True, thresh=-52.0, rest=-65.0, decay=1e-2)
    layers = {'X': inpt, 'E': exc, 'R': readout}
    
    for layer in layers:
        DQSNN.add_layer(layers[layer], name=layer)
    
    # Create Connections
    input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=torch.transpose(DQANN.fc1.weight, 0, 1) * scale1)
    exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=torch.transpose(DQANN.fc2.weight, 0, 1) * scale2)

    DQSNN.add_connection(input_exc_conn, source='X', target='E')

    if stdp == 0: # don't do connection updates
        DQSNN.learning = False
    else:
        w = torch.zeros((1000, 1000))
        if rand_init:
            w = torch.rand((1000, 1000))/100
            _ = w.as_strided([1000], [1001]).copy_(torch.zeros(1000))
    
    if stdp == 1: 
        # post_pre
        exc_exc_conn = Connection(source=layers['E'], target=layers['E'], w=w, 
                                  update_rule=post_pre, nu=(nu_pre, nu_post), norm=100)
        DQSNN.add_connection(exc_exc_conn, source='E', target='E')
    elif stdp == 2: 
        # wd_post_pre
        exc_exc_conn = Connection(source=layers['E'], target=layers['E'], w=w, 
                                  update_rule=wd_post_pre, nu=(nu_pre, nu_post), wmin=0, wmax=1)
        DQSNN.add_connection(exc_exc_conn, source='E', target='E')
    elif stdp == 3:
        exc_exc_conn = Connection(source=layers['E'], target=layers['E'], w=w, 
                                  update_rule=mstdp, nu=(nu_pre, nu_post))
        DQSNN.add_connection(exc_exc_conn, source='E', target='E')

    DQSNN.add_connection(exc_readout_conn, source='E', target='R')
    
    # Create Monitors.
    spikes = {}
    for layer in set(DQSNN.layers):
        spikes[layer] = Monitor(DQSNN.layers[layer], state_vars=['s'], time=runtime)
        DQSNN.add_monitor(spikes[layer], name='%s_spikes' % layer)
    
    return DQSNN
    

def main(dt=1.0, runtime=500, episodes=100, epsilon=0, device_id=0, **args):
    DQSNN = transfer('dqann.pt',
                     dt=dt, runtime=runtime, scale1=args['scale1'], scale2=args['scale2'], stdp=args['stdp'],
                     nu_pre=args['nu_pre'], nu_post=args['nu_post'], device_id=device_id, rand_init=args['rand_init'])
    
    ENV = GymEnvironment('BreakoutDeterministic-v4')
    ACTIONS = torch.tensor([0, 1, 2, 3])
    
    def policy(readout_spikes):
        q_values = torch.Tensor([readout_spikes[i].sum()
                                 for i in range(len(ACTIONS))])
        A = torch.ones(4) * epsilon / 4
        if torch.max(q_values) == 0:
            return torch.tensor([0.25, 0.25, 0.25, 0.25])
        best_action = torch.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    
    episode_rewards = torch.zeros(episodes)
    start_time = T()
    total_steps = 0
    
    if args['stdp'] > 0:
        masks = {('X','X'): torch.diag(torch.ones(1000))}
    
    if args['log']:
        curr_network_file = os.path.join(experiment_dir, f'{0:03d}_dqsnn.net')
        DQSNN.save(curr_network_file)
        print(f'Saved Start Network to {curr_network_file}.')
    
    for i_episode in range(episodes):
        obs = ENV.reset().cuda(device_id)
        state = torch.stack([obs] * 4, dim=2)
        last_reward = 0
        new_life = True
        prev_life = 5
        steps = 0
        
        for t in itertools.count():
            encoded_state = torch.tensor([0.25, 0.5, 0.75, 1]) * state
            encoded_state = torch.sum(encoded_state, dim=2)
            encoded_state = encoded_state.view([1, -1]).repeat(int(runtime/dt), 1)
            
            inpts = {'X': encoded_state}
            DQSNN.reset_()
            
            if args['stdp'] > 0:
                DQSNN.run(inpts, time=runtime, masks=masks, reward=last_reward)
            else:
                DQSNN.run(inpts, time=runtime)
            
            readout_spikes = DQSNN.monitors['R_spikes'].get('s')
            
            action_probs = policy(torch.sum(readout_spikes, dim=1))
            action = np.random.choice(ACTIONS, p=action_probs)
            
            if new_life:
                action = 1
            
            next_obs, reward, done, info = ENV.step(action)
            next_obs = next_obs.cuda(device_id)
            steps += 1
            if prev_life - info["ale.lives"] != 0:
                new_life = True
            else:
                new_life = False
            prev_life = info["ale.lives"]
    
            next_state = torch.clamp(next_obs - obs, min=0)
            next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
            
            if reward > 0:
                last_reward = 1
                episode_rewards[i_episode] += last_reward
                print(f'Episode: {i_episode + 1}, Step {t}, Reward {last_reward}, Total Reward {episode_rewards[i_episode]}')
                if args['stdp']:
                    print(f"Sum of weights in Lateral Connection {torch.sum(DQSNN.connections[('E', 'E')].w):.1f}")
            else:
                last_reward = 0
            
            state = next_state
            obs = next_obs
        
            if done:
                print(f'Step {steps} ({total_steps}) @ Episode {i_episode+1}/{episodes}')
                print(f'Episode Reward {episode_rewards[i_episode]}')
                sys.stdout.flush()
                break
        total_steps += steps
        
        if (i_episode + 1) % args['net_freeze_interval'] == 0 and args['stdp'] > 0 and args['log']:
            curr_network_file = os.path.join(experiment_dir, f'{i_episode + 1:03d}_dqsnn.net')
            DQSNN.save(curr_network_file)
            print(f'Saved Current Network to {curr_network_file}.')
        
    end_time = T()
    print(f'Average Reward over {episodes} Episodes {torch.sum(episode_rewards) / episodes}')
    if args['log']:
        viz_data = {'episode_rewards': episode_rewards}
        viz_data_file = os.path.join(experiment_dir, 'dqsnn.dat')
        curr_network_file = os.path.join(experiment_dir, f'{i_episode + 1:03d}_dqsnn.net')
        pickle.dump(viz_data, open(viz_data_file, 'wb'))
        DQSNN.save(curr_network_file)
        print(f'Saved Trained Network to {curr_network_file}.')
        print(f'Saved Collected Data to {viz_data_file}.')
    print(f'Total time taken: {end_time - start_time}')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--runtime', type=int, default=250)
    parser.add_argument('--scale1', type=float, default=6.452)
    parser.add_argument('--scale2', type=float, default=71.155)
    parser.add_argument('--rand_init', type=int, default=0)
    parser.add_argument('--stdp', type=int, default=0)
    parser.add_argument('--nu_pre', type=float, default=1e-7)
    parser.add_argument('--nu_post', type=float, default=1e-5)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--net_freeze_interval', type=int, default=5)
    parser.add_argument('--device_id', type=int, default=0)
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    if torch.cuda.is_available():
        torch.cuda.set_device(args['device_id'])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args['seed'])
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args['log']:
        try:
            os.mkdir(experiment_dir)
        except:
            pass
        class LogPrint(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)    
            def flush(self):
                for f in self.files:
                    f.flush()
            
        f = open(os.path.join(experiment_dir, 'dqsnn.log'), 'w')
        sys.stdout = LogPrint(sys.stdout, f)
    
    print(args)
    main(**args)
    