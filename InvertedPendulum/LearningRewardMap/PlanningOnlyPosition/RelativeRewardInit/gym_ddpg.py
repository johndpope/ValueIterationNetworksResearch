import time
import filter_env
from ddpg import *
import gc
from gym.wrappers import Monitor
from actor_network import ActorNetwork
gc.enable()

ENV_NAME = 'Pendulum-v0'
EPISODES = 2000
TEST = 50

def main():
    startTime = time.time()
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    results_file = open("DivergingReward.csv", 'a')
    agent = DDPG(env, results_file)
    env = Monitor(env, directory='experiments/' + ENV_NAME, force=True)
    results_file.write("Episodes Spent Training; " + str(TEST) + " Episode Eval Avg \n")
    for episode in range(EPISODES):
        state = env.reset()
        if (episode % 20 == 0):
            print("episode:",episode)
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
            results_file.write(str(episode) + "; " + str(ave_reward) + ";")
            results_file.write("Learned Reward Map; %s \n" % (np.array_str(agent.actor_network.net[-1].eval())).replace("\n", " "))

    results_file.write("Time Training (" + str(EPISODES) + "episodes);" + str(time.time() - startTime) + "\n")
    results_file.write("Learned Reward Map; %s \n" % (np.array_str(agent.actor_network.net[-1].eval())).replace("\n", " "))

    results_file.write("Evaluation Episode; Reward \n")
    # for episode in range(100):
    #     total_reward = 0
    #     state = env.reset()
    #     for j in range(env.spec.timestep_limit):
    #         action = agent.action(state) # direct action for test
    #         state,reward,done,_ = env.step(action)
    #         total_reward += reward
    #         if done:
    #             break
    #     results_file.write(str(episode) + "; " + str(total_reward) + "\n")
    results_file.write("endExperiment\n\n")
    results_file.close()

if __name__ == '__main__':
    main()
