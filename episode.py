from energia.env import *
from matplotlib import pyplot as plt

def episode(agent, verbose=False, epsilon=0.1, alpha=0.5, type='Sarsa'):

        type = 'Q-learning'

        temp_start_esterno = 15
        if verbose:
            print(" Episode: inizializzo esterno a temperatura {:2.1f}".format(temp_start_esterno), end="")
        esterno = Esterno(temp_start_esterno, verbose=verbose)
        if verbose:
            print(" ...fatto")

        temp_start_interno = 15

        if verbose:
            print(" Episode: inizializzo interno a temperatura {:2.1f}".format(temp_start_interno), end="")
        interno = Interno(temp_start_interno, verbose=verbose)
        if verbose:
            print(" ...fatto")

        temp_start_termosifone = 15
        if verbose:
            print(" Episode: inizializzo termosifone a temperatura {:2.1f}".format(temp_start_termosifone), end="")
        termosifone = Termosifone(esterno, interno, temp_start_termosifone, verbose=verbose)
        if verbose:
            print(" ...fatto")

        if verbose:
            print(" Episode: inizializzo Environment".format(temp_start_termosifone), end="")
        env = Environment(esterno, interno, termosifone, verbose=verbose)
        if verbose:
            print(" ...fatto")

        N_time_steps = 48
        if verbose:
            print(" Episode: Numero di time step per episodio = {:2d}".format(N_time_steps))

        tot_R = 0
        tot_gas = 0
        tot_n_pers = 0
        tot_a_caso = 0
        tot_a_caso_max = 0

        if verbose:
            print(' Episode: ---- Time Step = 0')
            print(' Episode: Calcolo lo stato di partenza')
        S_0 = env.calcola_stato()
        if verbose:
            print(' Episode: ...fatto')

        if verbose:
            print(' Episode: Scelgo la prima azione')

        azione_0 = agent.scegli_azione(S_0, 0, episode, epsilon=epsilon)
        a_caso = azione_0[3]
        a_caso_max = azione_0[4]
        azione_0 = azione_0[0:3]
        tot_a_caso += a_caso
        tot_a_caso_max += a_caso_max
        if verbose:
            print(" Episode: ...fatto")
        if verbose:
            input(' Episode: Invio per continuare')

        for t_step in range(1, N_time_steps + 1):

            if verbose:
                print(' Episode: ---- Time Step = {}'.format(t_step))
                print(' Episode: Modifico lo stato di Environment ')
            env.prossimo_stato(azione_0)

            if verbose:
                print(' Episode: Calcolo lo stato di Environment ')
            S_1 = env.calcola_stato()

            if verbose:
                print(' Episode: Calcolo il reward ')
            R, gas, n_pers = env.reward()
            if verbose:
                print(' Episode: Aggiono i totali ', end="")
            tot_R += R
            tot_gas += gas
            tot_n_pers += n_pers
            if verbose:
                print(" ...fatto")
                print(" Episode: Scelgo l'azione successiva")
            azione_1 = agent.scegli_azione(S_1, t_step, episode, epsilon=epsilon)
            a_caso = azione_1[3]
            a_caso_max = azione_1[4]
            azione_1 = azione_1[0:3]
            tot_a_caso += a_caso
            tot_a_caso_max += a_caso_max
            if verbose:
                print(" Episode: Eseguo il backup della variabile Q")
            Q_updated = agent.backup(t_step, R, S_0, S_1, azione_0, azione_1, alpha=alpha, type=type)
            if verbose:
                print(" Episode: Aggiorno lo stato e l'azione di partenza", end="")
            S_0 = S_1
            azione_0 = azione_1
            if verbose:
                print(" ...fatto")
            if verbose:
                input(' Episode: Invio per continuare')

        return env, tot_a_caso, tot_a_caso_max, tot_R, tot_gas, tot_n_pers

def episode_print(episode_i, env, tot_a_caso, tot_a_caso_max, tot_R, tot_gas, tot_n_pers):

    se_stampa_1 = episode_i < 20
    se_stampa_2 = episode_i < 100 and episode_i % 10 == 0
    se_stampa_3 = episode_i < 1000 and episode_i % 100 == 0
    se_stampa_4 = episode_i % 1000 == 0

    if se_stampa_1 or se_stampa_2 or se_stampa_3 or se_stampa_4:
        print("episode = {:4d}, tot_reward = {:7.1f}, gas = {:3.1f}, n_pers = {:4.1f}, n_pers_mean = {:2.1f}".format(
                episode_i, tot_R,
                tot_gas,
                tot_n_pers, tot_n_pers / 48))
        print('temp finale interno = {:2.2f}'.format(env.interno.temp))
        print('ho scelto a caso {:2d} volte, di cui {:2d} per max = 0'.format(tot_a_caso, tot_a_caso_max))

def episode_analysis(episode_i, env, interno,esterno,termosifone):

    fig1 = plt.figure(2)
    plt.plot(range(1, 50), interno.temp_seq, color='blue', label='temp interno')
    plt.plot(range(1, 49), termosifone.temp_su_seq, color='green', linestyle='--', label='temp su')
    plt.plot(range(1, 49), termosifone.temp_centro_seq, color='red', label='temp centro')
    plt.plot(range(1, 49), termosifone.temp_giu_seq, color='green', label='temp giu')
    plt.plot(range(1, 50), termosifone.acceso_seq, color='black', label='term acceso')

    plt.legend(loc='upper left')

    fig2 = plt.figure(3)

    plt.plot(range(1, 50), esterno.temp_seq, color='blue', figure=fig2, label='temp esterno')
    #mostra = int(input("Mostro la figura per l'episodio {}? (1 = si, 0 = no)".format(episode_i)))
    #if mostra == 1:
    plt.show()

def episode_optimal(agent, verbose=True, epsilon = 0.1, alpha = 0.5, ferma_time_step = False):

    tot_R = 0
    tot_gas = 0
    tot_n_pers = 0

    temp_start_esterno = 15
    if verbose:
        print(" Main: inizializzo esterno a temperatura {:2.1f}".format(temp_start_esterno), end="")
    esterno = Esterno(temp_start_esterno, verbose=verbose)
    if verbose:
        print(" ...fatto")

    temp_start_interno = 15
    if verbose:
        print(" Main: inizializzo interno a temperatura {:2.1f}".format(temp_start_interno), end="")
    interno = Interno(temp_start_interno, verbose=verbose)

    if verbose:
        print(" ...fatto")

    temp_start_termosifone = 15
    if verbose:
        print(" Main: inizializzo termosifone a temperatura {:2.1f}".format(temp_start_termosifone), end="")
    termosifone = Termosifone(esterno, interno, temp_start_termosifone, verbose=verbose)
    if verbose:
        print(" ...fatto")

    if verbose:
        print(" Main: inizializzo Environment".format(temp_start_termosifone), end="")
    env = Environment(esterno, interno, termosifone, verbose=verbose)
    if verbose:
        print(" ...fatto")

    N_time_steps = 48
    if verbose:
        print(' Episode: ---- Time Step = 0')
        print(' Episode: Calcolo lo stato di partenza')

    S_0 = env.calcola_stato()
    if verbose:
        print(' Episode: ...fatto')

    if verbose:
        print(' Episode: Scelgo la prima azione')

    azione_0 = agent.scegli_azione_optimal(S_0, 0)
    a_caso = azione_0[3]
    a_caso_max = azione_0[4]
    azione_0 = azione_0[0:3]
    if verbose:
        print(" Episode: ...fatto")
    if ferma_time_step:
        input(' Episode: Invio per continuare')

    for t_step in range(1, N_time_steps + 1):

        if verbose:
            print(' Episode: ---- Time Step = {}'.format(t_step))
            print(' Episode: Modifico lo stato di Environment ')
        env.prossimo_stato(azione_0)

        if verbose:
            print(' Episode: Calcolo lo stato di Environment ')
        S_1 = env.calcola_stato()

        if verbose:
            print(' Episode: Calcolo il reward ')
        R, gas, n_pers = env.reward()
        if verbose:
            print(' Episode: Aggiono i totali ', end="")
        tot_R += R
        tot_gas += gas
        tot_n_pers += n_pers
        if verbose:
            print(" ...fatto")
            print(" Episode: Scelgo l'azione successiva")
        azione_1 = agent.scegli_azione_optimal(S_1, t_step)
        a_caso = azione_1[3]
        a_caso_max = azione_1[4]
        azione_1 = azione_1[0:3]

        if verbose:
            print(" Episode: Eseguo il backup della variabile Q")
        Q_updated = agent.backup(t_step, R, S_0, S_1, azione_0, azione_1, alpha=alpha)
        if verbose:
            print(" Episode: Aggiorno lo stato e l'azione di partenza", end="")
        S_0 = S_1
        azione_0 = azione_1
        if verbose:
            print(" ...fatto")
        if ferma_time_step:
            input(' Episode: Invio per continuare')

    return env, interno,esterno,termosifone, interno.temp, tot_R, tot_gas, tot_n_pers


