import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib
import matplotlib.pyplot as plt

"""
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

"""

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs = 100, batch_size=5000, render=False):
    # make enviroment, check spaces, get obs/act dimensions
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n 

    # make a core policy network
    policy_net = mlp(sizes = [obs_dim] + hidden_sizes + [n_acts])

    # make function to compute action distribution
    
    def get_policy(obs):
        policy_nets = policy_net(obs)
        return Categorical(logits = policy_nets)

    # make action selection function (outputs int actions, sampled from policy)
    
    def get_action(obs):
        return get_policy(obs).sample().item()

    def reward_to_go(rews):
    
        rtgs = np.zeros_like(rews)
        n = len(rews)
        for i in reversed(range(n)):

            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)


        return rtgs


    # make loss function whose gradient, for the right data, is policy gradient
    
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp*weights).mean()

    # make optimizer
    optimizer = Adam(policy_net.parameters(), lr=lr)

    #for training policy

    def train_one_epoch(eppp):
        # make some empty lists for logging
        batch_obs = []  # for observations
        batch_acts = [] # for actions
        batch_weights = []  # for R(tau) wrighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measureing episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs coms from starting distribution
        done = False  # signal from env that episoce over
        ep_rews = []  # list fir rewards accrued throught ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        #collect experience by acting in the enviroment with current policy
        while True:
            
            # rendering 
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the enviroment
            act = get_action(torch.as_tensor(obs,dtype=torch.float32))
            obs, rew, done, _ = env.step(act)
            #print(obs)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                #print(ep_rews)
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) us R(tau)
                #batch_weights += [ep_ret]*ep_len
                
                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end expersirence loop if we have enough of it
                if len(batch_obs) > batch_size:
                    #print(batch_weights)
                    break
        
        # take a single policy gradient upate step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights,dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens, batch_size

    # training loop

    TotalEnvInterects = []
    Return_plot_mean = []
    Return_plot_std = []

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens, batch_size = train_one_epoch(i)

        Return_plot_mean.append(np.mean(batch_rets))
        Return_plot_std.append(np.std(batch_rets))
        TotalEnvInterects.append(i*batch_size)

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t'% 
             (i,batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


    # show the interaction
    obs = env.reset()
    while True:
        act = get_action(torch.as_tensor(obs,dtype=torch.float32))
        obs, rew, done, _ = env.step(act)
        env.render()
        if done:
            break

    # plotting
    under_line = np.array(Return_plot_mean) - np.array(Return_plot_std)
    upper_line = np.array(Return_plot_mean) + np.array(Return_plot_std)

    plt.plot(TotalEnvInterects,Return_plot_mean, color='green')
    plt.fill_between(TotalEnvInterects, upper_line, under_line, 
        alpha=0.15, color='green')
    plt.xlabel('TotalEnvInterects')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':

    env_name = 'CartPole-v1'
    hidden_sizes = [32]
    lr = 1e-2
    epochs = 60
    batch_size = 5000
    render = False

    train(env_name, hidden_sizes, lr, epochs, batch_size, render)
