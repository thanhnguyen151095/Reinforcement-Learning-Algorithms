import torch  
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
import time
import scipy.signal
from mpi4py import MPI
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size,obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self,batch_size=32):
        idxs = np.random.randint(0,self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def mlp(sizes, activation, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):

    def __init__(self,obs_dim,act_dim,hidden_sizes,activation,act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self,obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self,obs_dim, act_dim, hidden_sizes,activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=[256]+[256], 
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy
        self.pi = MLPActor(obs_dim,act_dim,hidden_sizes,activation,act_limit)
        # build value functions    
        self.q1  = MLPQFunction(obs_dim,act_dim,hidden_sizes,activation)
        self.q2  = MLPQFunction(obs_dim,act_dim,hidden_sizes,activation)

    def act(self,obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


def td3(env_name='InvertedPendulum-v2',hidden_sizes = [256] + [256], seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1,target_noise=0.2,
         noise_clip=0.5,policy_delay=2,num_test_episodes=10, 
         max_ep_len=1000, save_freq=1):

    # Random seed   
    torch.manual_seed(seed)
    np.random.seed(seed)

    # make enviroment, check spaces, get obs/act dimensions
    env = gym.make(env_name)
    test_env = env

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Set up experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q function
        with torch.no_grad():

            # Target policy smoothing
            a2 = ac_targ.pi(o2) 
            ns = torch.randn_like(a2) * target_noise
            ns = torch.clamp(ns, -noise_clip, noise_clip)
            a2 = torch.clamp(a2 + ns, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2,a2)
            q2_pi_targ = ac_targ.q2(o2,a2)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1- backup)**2).mean()
        loss_q2 = ((q2- backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o,ac.pi(o))
        return -q1_pi.mean()

    # make optimizer for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr = pi_lr)
    q_optimizer = Adam(q_params, lr = q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.

            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfeeze Q-network so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaginf.
            with torch.no_grad():
                for p, p_targ in  zip(ac.parameters(),ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    #  params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1-polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o,dtype=torch.float32))
        a += noise_scale*np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        Return_plot_mean = []
        Return_plot_std = []
        ep_return = []
        for k in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o,r,d, _ = test_env.step(get_action(o,0))
                ep_ret += r
                ep_len += 1
            ep_return.append(ep_ret)

        ep_return_mean = np.mean(ep_return)
        ep_return_std = np.std(ep_return)

        return ep_return_mean, ep_return_std


    # Prepare for interaction with enviroment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    start_time = time.time()


    # Main loop: collect experience in env and update/log each epoch
    Return_plot_mean = []
    Return_plot_std = []
    TotalEnvInterects = []
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        #env.render()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # save 
        replay_buffer.store(o, a, r, o2, d)

        # update obs
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t>= update_after and t% update_every ==0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch,timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            ep_rt_mean, ep_rt_std = test_agent()
            Return_plot_mean.append(ep_rt_mean)
            Return_plot_std.append(ep_rt_std)

            print('Epoch: %3d \t  Time: %.2f \t return: %.3f \t'%((t+1)/steps_per_epoch, 
                time.time()-start_time,ep_rt_mean))

            TotalEnvInterects.append(t)

    # save a policy model
    torch.save(ac.pi.state_dict(), '/home/win/spinningup/spinup/code_NGUYEN/pi_td3.pth')

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

    td3(env_name='InvertedPendulum-v2',hidden_sizes = [256] + [256], seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
         noise_clip=0.5,policy_delay=2,num_test_episodes=10, 
         max_ep_len=1000, save_freq=1)

   
