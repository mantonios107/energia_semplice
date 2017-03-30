def domanda_input():

    #n_episodes = 100

    trovato = [0,0]

    while trovato != [1,1]:
        verbose_domanda = input('Vuoi le stampe per debug ad ogni passo?(1 = si, 0 = no, invio = default = 0) Dopo quanti time steps? (invio = default = 100)')

        if verbose_domanda != '':
            verbose_domanda = verbose_domanda.split(sep = ' ')

            try:
                verbose = int(verbose_domanda[0])
                if verbose == 1:
                    verbose = True
                    trovato[0] = 1
                elif verbose == 0:
                    verbose = False
                    trovato[0] = 1
                else:
                    print('Primo numero inserito non valido')
                if len(verbose_domanda) != 1:
                    if verbose_domanda[1] != '':
                        n_episodes = int(verbose_domanda[1])
                        trovato[1] = 1
                    else:
                        trovato[1] = 1

            except ValueError:
                print('Uno fra i numeri inseriti non è un intero')
        else:
            trovato = [1,1]
            verbose = False
            n_episodes = 100

        while True:
            eps_domanda = input('Inserisci epsilon tra 0 e 1 (invio = default (0.1) )')
            if eps_domanda != '':
                try:
                    eps_domanda = float(eps_domanda)
                    if 1 >= eps_domanda >= 0:
                        epsilon = eps_domanda
                        break
                    else:
                        print('Valore float epsilon inserito non valido')
                except ValueError:
                    print('Valore epsilon inserito non è un float')
            else:
                epsilon = 0.1
                break

        while True:
            alpha_domanda = input('Inserisci alpha tra 0 e 1 (invio = default (0.5) )')
            if alpha_domanda != '':
                try:
                    alpha_domanda = float(alpha_domanda)
                    if 1 >= alpha_domanda >= 0:
                        alpha = alpha_domanda
                        break
                    else:
                        print('Valore float alpha inserito non valido')
                except ValueError:
                    print('Valore alpha inserito non è un float')
            else:
                alpha = 0.5
                break

    return verbose, n_episodes, epsilon, alpha