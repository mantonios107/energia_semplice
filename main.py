from energia.agent import Agent
from energia.env import *
from matplotlib import pyplot as plt
import pandas as pd
from domanda_input import domanda_input
from episode import *

if __name__ == '__main__':

    verbose, n_episodes, epsilon, alpha = domanda_input()

    R_sequence = np.zeros(n_episodes)
    gas_tot_sequence = np.zeros(n_episodes)
    n_pers_tot_sequence = np.zeros(n_episodes)
    a_caso_sequence = np.zeros(n_episodes)

    agent = Agent(verbose=verbose)

    for episode_i in range(1, n_episodes + 1):

        env, tot_a_caso, tot_a_caso_max, tot_R, tot_gas, tot_n_pers = episode(agent, verbose, epsilon, alpha)
        episode_print(episode_i, env, tot_a_caso, tot_a_caso_max, tot_R, tot_gas, tot_n_pers)

        R_sequence[episode_i - 1] = tot_R
        gas_tot_sequence[episode_i - 1] = tot_gas
        n_pers_tot_sequence[episode_i - 1] = tot_n_pers
        a_caso_sequence[episode_i - 1] = tot_a_caso

        if verbose:
            episode_analysis(episode_i, env)

    print('')

    episode_i +=1

    verbose_optimal = input('Vuoi le stampe della policy ottimale? (1 = si, 0 = no, invio = default = 0)')
    ferma_time_step = input('Vuoi fermare ad ogni time step? (1 = si, 0 = no, invio = default = 0)')

    if int(verbose_optimal) == 1:
        verbose_optimal = True
    else:
        verbose_optimal = False
    if int(ferma_time_step) == 1:
        ferma_time_step = True
    else:
        ferma_time_step = False

    env_optimal, interno_optimal, esterno_optimal, termosifone_optimal, temp_fine_interno, tot_R, tot_gas, tot_n_pers = episode_optimal(agent, verbose=verbose_optimal, ferma_time_step=ferma_time_step)

    print("episode = {:4d}, tot_reward = {:7.1f}, gas = {:3.1f}, n_pers = {:4.1f}, n_pers_mean = {:2.1f}".format(
            episode_i, tot_R, tot_gas, tot_n_pers, tot_n_pers / 48))
    print('temp finale interno = {:2.2f}'.format(temp_fine_interno))

    plot_optimal = input(' Vuoi i plot della policy ottimale? (1 = si, 0 = no, invio = default = 0)')
    if int(plot_optimal) == 1:
        episode_analysis(episode_i, env_optimal, interno_optimal, esterno_optimal, termosifone_optimal)


    # intervallo filtro a media mobile
    N = 200

    plt.figure(4)
    plt.plot(range(1, n_episodes + 1), R_sequence, color='blue')
    R_sequence_filtered = pd.rolling_mean(R_sequence, N, min_periods=0, center=True)
    plt.plot(range(1, n_episodes + 1), R_sequence_filtered, color='red')
    plt.legend()

    plt.figure(5)
    plt.plot(range(1, n_episodes + 1), gas_tot_sequence, color='blue', label='gas_tot_sequence')
    gas_tot_sequence_filtered = pd.rolling_mean(gas_tot_sequence, N, min_periods=0, center=True)
    plt.plot(range(1, n_episodes + 1), gas_tot_sequence_filtered, color='red')
    plt.legend()

    plt.figure(6)
    plt.plot(range(1, n_episodes + 1), n_pers_tot_sequence, color='blue', label='n_pers_tot_sequence')
    n_pers_tot_sequence_filtered = pd.rolling_mean(n_pers_tot_sequence, N, min_periods=0, center=True)
    plt.plot(range(1, n_episodes + 1), n_pers_tot_sequence_filtered, color='red')
    plt.legend()

    plt.show()

