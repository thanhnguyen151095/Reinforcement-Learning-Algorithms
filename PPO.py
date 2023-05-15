import torch  
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
import time
from gym.spaces import Discrete, Box
import scipy.signal
from mpi4py import MPI
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import matplotlib
import matplotlib.pyplot as plt



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mpi_statistics_scalar(x, with_min_and_max=False):

    def allreduce(*args, **kwargs):
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

    def mpi_op(x, op):
        x, scalar = ([x], True) if np.isscalar(x) else (x, False)
        x = np.asarray(x, dtype=np.float32)
        buff = np.zeros_like(x, dtype=np.float32)
        allreduce(x, buff, op=op)
        return buff[0] if scalar else buff

    def mpi_sum(x):
        return mpi_op(x, MPI.SUM)

    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std

class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam = 0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        #if condition returns True, then nothing happens:if condition returns False, AssertionError is raised:
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {kk: torch.as_tensor(vv, dtype=torch.float32) for kk,vv in data.items()}


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def PPO(env_name='CartPole-v1', hidden_sizes= [64]+[64],seed = 0, steps_per_epoch = 4000, epochs = 50, gamma = 0.99,
        clip_ratio = 0.2, pi_lr=3e-4,v_lr = 1e-3 , train_pi_iters = 80, train_v_iters = 80,
        lam = 0.97, max_ep_len = 1000, target_kl = 0.01, render=False):
        
    
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed   
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # make enviroment, check spaces, get obs/act dimensions
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    act_dim = env.action_space.shape


    # make a core policy network
    policy_net = mlp(sizes = [obs_dim] + hidden_sizes + [n_acts])

    # build  a value function network
    value_net  = mlp(sizes = [obs_dim] + hidden_sizes + [1])

    # Sync params across processes
    #sync_params((policy_net , value_net))

    sync_params(policy_net)
    sync_params(value_net)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs()) # divided by the number of processors
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    
    def policy_value_nets_step(obs):
        with torch.no_grad():
            pi = Categorical(logits = policy_net(obs))
            act = pi.sample()
            logp_a = pi.log_prob(act)

            v = torch.squeeze(value_net(obs),-1) # Critical to ensure v has right shape.


        return act.numpy(), logp_a.numpy(), v.numpy()

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi = Categorical(logits = policy_net(obs))
        logp = pi.log_prob(act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped,dtype = torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent = ent, cf= clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((torch.squeeze(value_net(obs),-1) - ret)**2).mean()

    # make optimizer
    pi_optimizer = Adam(policy_net.parameters(), lr = pi_lr)
    v_optimizer = Adam(value_net.parameters(), lr = v_lr)

    # Prepare for interaction with enviroment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    Return_plot_mean = []
    Return_plot_std = []
    TotalEnvInterects = []
    start_time = time.time()
    for epoch in range (epochs):
        ep_return = []
        ep_lenght = []
        for t in range(local_steps_per_epoch):
            #env.render()

            a, logp, v = policy_value_nets_step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, done, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save 
            buf.store(o, a, r, v, logp)

            # update obs
            o = next_o

            timeout = ep_len == max_ep_len # check the epoch len == the maximum of epoch limitatio 1000
            terminal = done or timeout
            epoch_ended = t == local_steps_per_epoch - 1 # check the batch size 

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                # if trajectory didn't reach the terminal state, bootstrap value target 
                if timeout or epoch_ended:
                    _, _, v = policy_value_nets_step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0 
                buf.finish_path(v)
                if terminal:
                    ep_return.append(ep_ret)
                    ep_lenght.append(ep_len)

                o, ep_ret, ep_len = env.reset(), 0, 0

        
        # Perform PPO update!
        data =  buf.get()
        loss_pi_old, pi_info_old = compute_loss_pi(data)
        loss_v_old = compute_loss_v(data)

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5*target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(policy_net) # average grads across MPI processes
            pi_optimizer.step()

        
        # Value function learning
        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(value_net) # average grads across MPI processes
            v_optimizer.step()


        print('Epoch: %3d \t  Time: %.2f \t   loss_pi: %f \t loss_v: %.3f \t return: %.3f \t ep_len: %.3f'%(epoch, 
                time.time()-start_time, loss_pi_old, loss_v_old,np.mean(ep_return), np.mean(ep_lenght)))
        
        Return_plot_mean.append(np.mean(ep_return))
        Return_plot_std.append(np.std(ep_return))
        TotalEnvInterects.append(epoch*local_steps_per_epoch)

    # save a policy model
    torch.save(policy_net.state_dict(), '/home/win/spinningup/spinup/examples/pytorch/pg_math/pi_ppo.pth')
    torch.save(value_net.state_dict(), '/home/win/spinningup/spinup/examples/pytorch/pg_math/v_ppo.pth')
    
    # show the interaction
    obs = env.reset()
    while True:
        act, _, _ = policy_value_nets_step(torch.as_tensor(obs, dtype=torch.float32))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    env_name = 'CartPole-v1'
    hidden_sizes = [64] + [64]
    seed = 0
    steps_per_epoch = 4000
    epochs = 50 
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4 #3e-4
    v_lr = 1e-3 
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97 
    max_ep_len = 1000
    target_kl = 0.01
    render=False

    PPO(env_name, hidden_sizes,seed, steps_per_epoch, epochs, 
        gamma , clip_ratio , pi_lr,v_lr , train_pi_iters, 
        train_v_iters, lam, max_ep_len, target_kl, render)
    
