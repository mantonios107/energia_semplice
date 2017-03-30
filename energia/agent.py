import numpy as np
from scipy.special import binom

class Agent:

    def __init__(self, verbose: bool = False) -> None:

        # discretizzo la temperatura ogni 0.5 C
        #  12h / (15min) = 48 time steps ogni episodio, esterno (0 - 40), interno (15 - 30),azioni (15 - 30,  31*30*29/6)

        self.dim = 81, 31, 465 # 48 time steps
        self.Q = np.zeros(self.dim)
        if verbose:
            self.verbose = True
            self.azione_seq = []
            self.scelta_seq = [] # 0 a caso, 1 scelgo il index_max_Q
        else:
            self.verbose = False

    def cerca_max_Q(self, t_step, indice_esterno, indice_interno):

        index = np.zeros((1))
        index = self.Q
        index = np.argmax(self.Q[ indice_esterno,indice_interno,:])
        max = np.max(self.Q[ indice_esterno, indice_interno,:])
        if self.verbose:
            print('  Agent.cerca_max_Q:  t_step, indice_esterno, indice_interno, index, max = {:2d} {:2d} {:2d} {:2d} {:3.3f}'.format(t_step, indice_esterno, indice_interno, index, max))

        return max, index

    def converti_azione_in_num_465(self, su, centro, giu):
        centro, giu = centro - 15, giu -15
        centro, giu = centro * 2 + 1, giu * 2 + 1

        num = binom(centro - 1, 2) + giu - 1

        assert (num < 466)

        return int(num)

    def converti_azione_in_num_4495(self, su, centro, giu):
        # mappa dalla terna su,centro,giu a Z_(31 3) binomiale seguendo lo schema
        # 321
        #
        # 421
        # 431 432
        #
        # 521
        # 531 532
        # 541 542 543 etc.

        # formula, a>b>c: (a b c ) --> (a-1 3) binomiale + (b-1 2) binomiale + (c-1 1) binomiale

        # converto dall'intervallo [15,30] discretizzato a 0.5 all'intervallo [1,31]
        su, centro, giu = su - 15, centro - 15, giu - 15

        su, centro, giu = su * 2 + 1 , centro * 2 + 1 , giu*2 + 1

        num = binom(su-1,3)  +  binom(centro-1,2) + giu - 1

        assert(num < 4496)
        return int(num)

    def converti_num_in_azione_465(self, num):

        su = 30

        for centro in range(2, 31):
            if num < binom(centro,2):
                break
        centro -= 1
        num -= binom(centro, 2)

        for giu in range(1, 30):
            if (num < binom(giu, 1)):
                break
        giu -= 1
        num -= giu

        centro, giu = centro / 2 + 15, giu / 2 + 15
        if self.verbose:
            print('  Agent.converti_num_in_azione_465: su, centro, giu = {} {} {}'.format(su, centro, giu))
        return su, centro, giu, num

    def converti_num_in_azione_4495(self,num):

        for su in range(3, 32):
            if (num < binom(su, 3)):
                break
        su -= 1
        num -= binom(su, 3)

        for centro in range(2, 31):
            if num < binom(centro,2):
                break
        centro -= 1
        num -= binom(centro, 2)

        for giu in range(1, 30):
            if (num < binom(giu, 1)):
                break
        giu -= 1
        num -= giu

        su, centro, giu = su/2 + 15, centro/2 + 15, giu/2 + 15
        if self.verbose:
            print('  Agent.converti_num_in_azione_4495: su, centro, giu = {} {} {}'.format(su,centro,giu))
        return su,centro,giu,num

    def converti_stato(self,stato):

        indice_esterno = int(stato[0] * 2)
        indice_interno = int((stato[1]-15) * 2)
        return indice_esterno, indice_interno

    def scegli_azione(self, stato, t_step, episode, epsilon = 0.1):

        # u variabile di scelta per policy epsilon-greedy
        indice_esterno,indice_interno = self.converti_stato(stato)
        u = np.random.rand(1)
        max, index = self.cerca_max_Q(t_step, indice_esterno, indice_interno)
        a_caso = 0
        a_caso_max = 0
        if (max == 0) or (u < epsilon):

            # scelgo un'azione casuale
            num = np.random.randint(1,465)

            # converto in tupla
            azione = self.converti_num_in_azione_465(num)

            #if azione[3] != 0:
            #    print('ERROR: num diverso da 0 = {}'.format(num))
            su,giu,centro = azione[0:3]

            #print('max, u = {} {}'.format(max, u) )
            a_caso = 1
            if max == 0:
                a_caso_max = 1

            if self.verbose:
                if max == 0:
                    print('  Agent.scegli_azione: scelgo a caso perché max == 0')
                else:
                    print('  Agent.scegli_azione: scelgo a caso poiché u = {:1.3f} < epsilon = {:1.3f}'.format(u[0],epsilon))
                self.azione_seq.append((su,giu,centro))
                print('  Agent.scegli_azione: num = {}, convertito in azione e num finale = {}'.format(num, azione))

            return su,giu,centro, a_caso,a_caso_max

        else:

            # scelgo azione con miglior Q per questo time step
            su,centro,giu,num = self.converti_num_in_azione_465(index)
            if self.verbose:
                print('  Agent.scegli_azione: scelgo azione con miglior Q per questo time step: u = {}, index = {}'.format(u, index))

            return su, centro, giu, a_caso, a_caso_max

    def backup(self, t_step, reward, stato_0, stato_1, azione_0, azione_1, alpha=0.5, gamma=0.99, type = 'Sarsa'):

        if self.verbose:
            print("  Agent.backup: stato_0 = {} , stato_1 = {}".format(stato_0, stato_1))
            print("  Agent.backup: azione_0 = {} , azione_1 = {}".format(azione_0, azione_1))
        indice_esterno_0, indice_interno_0 = self.converti_stato(stato_0)
        indice_esterno_1, indice_interno_1 = self.converti_stato(stato_1)
        indice_azione_0 = self.converti_azione_in_num_465(azione_0[0], azione_0[1], azione_0[2])
        indice_azione_1 = self.converti_azione_in_num_465(azione_1[0], azione_1[1], azione_1[2])

        if type == 'Sarsa':

            if t_step < 48:
                if self.verbose:
                    Q_prima = self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]
                    Q_seguente = self.Q[ indice_esterno_1, indice_interno_1, indice_azione_1]

                self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0] += alpha * (reward + gamma * self.Q[ indice_esterno_1, indice_interno_1, indice_azione_1] - self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0])

                if self.verbose:
                    print('  Agent.backup: t_step, indice_esterno_0, indice_interno_0, indice_azione_0 = {:2d} {:3d} {:3d} {:4d}'.format(t_step, indice_esterno_0, indice_interno_0, indice_azione_0))
                    print('  Agent.backup: Q_prima = {:4.2f}'.format(Q_prima))
                    print('  Agent.backup: Q_seguente = {:4.2f} '.format(Q_seguente))
                    print('  Agent.backup: Q_dopo = {:4.2f}'.format(self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]))

                return self.Q[ indice_esterno_0,indice_interno_0,indice_azione_0]
            else:
                if self.verbose:
                    Q_prima = self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]

                self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0] += alpha * (
                reward - self.Q[
                     indice_esterno_0, indice_interno_0, indice_azione_0])

                if self.verbose:
                    print('  Agent.backup: t_step, indice_esterno_0, indice_interno_0, indice_azione_0 = {:2d} {:3d} {:3d} {:4d}'.format(t_step, indice_esterno_0, indice_interno_0, indice_azione_0))
                    print('  Agent.backup: Q_prima = {:4.2f}'.format(Q_prima))
                    print('  Agent.backup: non alloco Q_seguente')
                    print('  Agent.backup: Q_dopo = {:4.2f}'.format(self.Q[t_step, indice_esterno_0, indice_interno_0, indice_azione_0]))

                return self.Q[  indice_esterno_0, indice_interno_0, indice_azione_0]

        if type == 'Q-learning':

            if t_step < 48:

                Q_max_a = np.max(self.Q[ indice_esterno_1, indice_interno_1, :])
                self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0] += alpha * (reward + gamma * Q_max_a - self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0])

                if self.verbose:
                    Q_prima = self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]
                    Q_seguente = Q_max_a

                if self.verbose:
                    print('  Agent.backup: t_step, indice_esterno_0, indice_interno_0, indice_azione_0 = {:2d} {:3d} {:3d} {:4d}'.format(t_step, indice_esterno_0, indice_interno_0, indice_azione_0))
                    print('  Agent.backup: Q_prima = {:4.2f}'.format(Q_prima))
                    print('  Agent.backup: Q_seguente = {:4.2f} '.format(Q_seguente))
                    print('  Agent.backup: Q_dopo = {:4.2f}'.format(self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]))

                return self.Q[ indice_esterno_0,indice_interno_0,indice_azione_0]
            else:
                if self.verbose:
                    Q_prima = self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0]

                self.Q[ indice_esterno_0, indice_interno_0, indice_azione_0] += alpha * (
                reward - self.Q[
                     indice_esterno_0, indice_interno_0, indice_azione_0])

                if self.verbose:
                    print('  Agent.backup: t_step, indice_esterno_0, indice_interno_0, indice_azione_0 = {:2d} {:3d} {:3d} {:4d}'.format(t_step, indice_esterno_0, indice_interno_0, indice_azione_0))
                    print('  Agent.backup: Q_prima = {:4.2f}'.format(Q_prima))
                    print('  Agent.backup: non alloco Q_seguente')
                    print('  Agent.backup: Q_dopo = {:4.2f}'.format(self.Q[indice_esterno_0, indice_interno_0, indice_azione_0]))

                return self.Q[  indice_esterno_0, indice_interno_0, indice_azione_0]


    def scegli_azione_optimal(self, stato, t_step):

        indice_esterno, indice_interno = self.converti_stato(stato)
        u = np.random.rand(1)
        max, index = self.cerca_max_Q(t_step, indice_esterno, indice_interno)

        su, centro, giu, num = self.converti_num_in_azione_465(index)
        if self.verbose:
            print('  Agent.scegli_azione_optimal: su, centro, giu = {} {} {}'.format(su, centro, giu))

        a_caso = 0
        a_caso_max = 0

        return su, centro, giu, a_caso, a_caso_max
