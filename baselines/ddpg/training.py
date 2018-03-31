import copy
import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG, clear_path
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, nb_eval_test_steps, batch_size, memory,
    tau=0.01, train_eval_env=None, test_eval_env=None, param_noise_adaption_interval=50, learning_steps=1, window_length=50,
    eval_period=50, infer_path='./infer/', summary_dir='./results/'):
    rank = MPI.COMM_WORLD.Get_rank()

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, learning_steps=learning_steps, summary_dir=summary_dir)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    if not os.path.exists(os.path.join(infer_path, 'train/')):
        os.makedirs(os.path.join(infer_path, 'train/'), exist_ok=True)
    else:
        clear_path(os.path.join(infer_path, 'train/'))

    if not os.path.exists(os.path.join(infer_path, 'test/')):
        os.makedirs(os.path.join(infer_path, 'test/'), exist_ok=True)
    else:
        clear_path(os.path.join(infer_path, 'test/'))


    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        train_summary_ops, train_summary_vars = agent.build_train_summaries()
        episode_summary_ops, episode_summary_vars = agent.build_episode_summaries()
        sess.graph.finalize()

        agent.reset()

        overall_t_train = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                episode_rollout = deque()

                episode_reward = 0
                obs, info = env.reset()
                obs = np.concatenate((obs['obs'].flatten(), obs['weights']))

                episode_rollout.append(obs)
                for rollout_step in range(learning_steps - 1):
                    obs = episode_rollout[-1]
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    new_obs, reward, done, info = env.step(action)
                    new_obs = np.concatenate((new_obs['obs'].flatten(), new_obs['weights']))
                    [episode_rollout.append(item) for item in [action, reward, done, info['next_y1'], new_obs]]
                    episode_reward += reward

                for t_rollout in range(nb_rollout_steps):
                    #print("HERE:", episode_rollout[-1])
                    action, q = agent.pi(episode_rollout[-1], apply_noise=True, compute_Q=True)
                    new_obs, reward, done, info = env.step(action)
                    new_obs = np.concatenate((new_obs['obs'].flatten(), new_obs['weights']))
                    [episode_rollout.append(item) for item in [action, reward, done, info['next_y1'], new_obs]]

                    agent.store_transition(copy.copy(episode_rollout))
                    [episode_rollout.popleft() for _ in range(5)]

                    episode_reward += reward

                    if done or t_rollout == nb_rollout_steps - 1:
                        summary = sess.run(episode_summary_ops, feed_dict={
                                episode_summary_vars[0]: episode_reward
                            })
                        agent.writer.add_summary(summary, epoch*nb_epoch_cycles + cycle)
                        agent.writer.flush()
                        break

                for t_train in range(nb_train_steps):
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        #epoch_adaptive_distances.append(distance)

                    cl, al, add_loss = agent.train()
                    summary = sess.run(train_summary_ops, feed_dict={
                                                train_summary_vars[0]: al,
                                                train_summary_vars[1]: cl,
                                                train_summary_vars[2]: add_loss
                                            })
                    agent.writer.add_summary(summary, overall_t_train)
                    overall_t_train += 1
                    #epoch_critic_losses.append(cl)
                    #epoch_actor_losses.append(al)
                    agent.update_target_net()                    

                if (epoch*nb_epoch_cycles + cycle) % eval_period == 0:
                    # infer
                    infer(agent, epoch*nb_epoch_cycles + cycle, train_eval_env, learning_steps, os.path.join(infer_path, 'train/'))
                    infer(agent, epoch*nb_epoch_cycles + cycle, test_eval_env, learning_steps, os.path.join(infer_path, 'test/'))

def infer(agent, episode, env, learning_steps, save_path):
    episode_rollout = deque()
    obs, info = env.reset()
    #print("OBS:", obs)
    obs = np.concatenate((obs['obs'].flatten(), obs['weights']))
    episode_rollout.append(obs)

    episode_reward = 0

    for rollout_step in range(learning_steps - 1):
        obs = episode_rollout[-1]
        action, _ = agent.pi(obs, apply_noise=False, compute_Q=False)
        new_obs, reward, done, info = env.step(action)
        new_obs = np.concatenate((new_obs['obs'].flatten(), new_obs['weights']))
        [episode_rollout.append(item) for item in [action, reward, done, info['next_y1'], new_obs]]
        episode_reward += reward

    for t_rollout in range(env.steps - learning_steps):
        action, _ = agent.pi(episode_rollout[-1], apply_noise=False, compute_Q=False)
        new_obs, reward, done, info = env.step(action)
        new_obs = np.concatenate((new_obs['obs'].flatten(), new_obs['weights']))
        [episode_rollout.append(item) for item in [action, reward, done, info['next_y1'], new_obs]]

        agent.store_transition(copy.copy(episode_rollout))
        [episode_rollout.popleft() for _ in range(5)]

        episode_reward += reward

        if done or t_rollout == env.steps - learning_steps - 1:
            break

    env.render()
    plt.savefig(os.path.join(save_path, str(episode)+".png"))
    plt.close()
