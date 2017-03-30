import numpy as np
from matplotlib import pyplot

class Termosifone:

    def __init__(self, esterno, interno, temp_propria, acceso = False, verbose = False):
        self.esterno = esterno
        self.interno = interno
        self.temp_propria = temp_propria
        self.acceso = acceso

        if verbose:
            self.verbose = True
            self.temp_seq = [] #np.zeros((48))
            self.temp_su_seq = [] #np.zeros((48))
            self.temp_centro_seq = [] #np.zeros((48))
            self.temp_giu_seq = [] #np.zeros((48))
            if self.acceso:
                self.acceso_seq = [1]
            else:
                self.acceso_seq = [0]
        else:
            self.verbose = False

    def set_points(self, su, giu, centro):
        self.term_su, self.term_giu, self.term_centro = su, giu, centro

        if self.verbose:
            self.temp_su_seq.append(su)
            self.temp_centro_seq.append(centro)
            self.temp_giu_seq.append(giu)

    def termostato(self):
        #acceso booleano
        if self.verbose:
            acceso_prima = self.acceso

        if self.interno.temp < self.term_giu:
            self.acceso = True
        elif self.interno.temp > self.term_centro:
            self.acceso = False

        if self.verbose:
            if(self.acceso == True):
                print('  Termosifone.termostato: il termosifone è acceso')
                self.acceso_seq.append(1)
            else:
                print('  Termosifone.termostato: il termosifone è spento')
                self.acceso_seq.append(0)

            if self.acceso == acceso_prima:
                print('  Termosifone.termostato: il termosifone rimane uguale a come era prima')
            else:
                print('  Termosifone.termostato: il termosifone ha cambiato il suo stato')


        self.modifica_temp_propria()

    def modifica_temp_propria(self):

        if not(self.acceso):
            self.temp_propria = self.esterno.temp
            #fattore_influenza_esterno = 1.2
            #self.temp_propria += (esterno.temp - self.temp_propria)/fattore_influenza_esterno
        else:
            self.temp_propria = 55
            #max_temp = 50
            #fattore_influenza_max_temp = 1.2
            #self.temp_propria += (max_temp - self.temp_propria) / fattore_influenza_max_temp


class Esterno:

    def __init__(self, temp, stagione = None, umidita=None, verbose = False):
        self.stagione = stagione
        self.temp = temp
        self.umidita = umidita
        if verbose:
            self.verbose = True
            self.temp_seq = [temp]
        else:
            self.verbose = False

    def modifica_temp(self,temp):
        self.temp = temp
        if self.verbose:
            print('  Esterno.modifica_temp: temperatura esterna = {:2.2f}'.format(self.temp))
            self.temp_seq.append(temp)

class Interno:

    def __init__(self,temp,umidita = None, verbose = False):
        self.temp = temp
        self.umidita = umidita
        if verbose:
            self.verbose = True
            self.temp_seq = [temp]
        else:
            self.verbose = False

    def modifica_temp(self,temp):
        self.temp = temp
        if self.verbose:
            print('  Interno.modifica_temp: temperatura interna = {:2.2f} '.format(temp))
            self.temp_seq.append(temp)

class Environment:

    def __init__(self,esterno,interno,termosifone, verbose = True):
        self.interno = interno
        self.esterno = esterno
        self.termosifone = termosifone

        if verbose:
            self.verbose = True
            self.n_pers_seq = []
            self.malessere_seq = []
            self.reward_seq = []
            self.stato_seq = []
        else:
            self.verbose = False

    def calcola_malessere(self):
        temp_obiettivo = 22  # potenzialmente funzione dell'esterno
        umidita_obiettivo = 50  # potenzialmente funzione dell'esterno
        fattore_temp_umidita = 10  # bla bla...
        malessere = abs(self.interno.temp - temp_obiettivo) #+ (self.interno.umidita - umidita_obiettivo) / fattore_temp_umidita
        if self.verbose:
            self.malessere_seq.append(malessere)
            print('  Environment.calcola_malessere: malessere = {:2.3f}'.format(malessere))
        return malessere

    def calcola_pers(self,malessere):
        n_pers_standard = 80
        # n_pers_delta = np.random.poisson(min(5 * malessere, 100))
        n_pers_delta = 15 * malessere
        n_pers = n_pers_standard - n_pers_delta
        if self.verbose:
            self.n_pers_seq.append(n_pers)
            print('  Environment.calcola_pers: media  = {:2.1f}'.format(min(1 / malessere, 100)))
            print('  Environment.calcola_pers: n_pers = {:3d}'.format(int(n_pers)))

        return n_pers

    def reward(self, terminale = False):

        malessere = self.calcola_malessere()
        n_pers = self.calcola_pers(malessere)

        # 10 € ogni 15 min di accensione termosifone
        if self.termosifone.acceso == True:
            gas = 10
        else:
            gas = 0

        soldi_per_pers = 50
        const_gas = 1
        const_el = 0.5
        elettricita = 2

        J = (n_pers * soldi_per_pers) - (gas * const_gas + elettricita * const_el)

        if self.verbose:
            self.reward_seq.append(J)
            print('  Environment.reward: reward = {:3.2f}'.format(J))
        return J, gas, n_pers

    def modifica_interno(self):

        fattore_adattamento_esterno = 10
        nuova_temp_interno = self.interno.temp
        adatta_a_esterno = (self.esterno.temp - self.interno.temp) / fattore_adattamento_esterno
        nuova_temp_interno += adatta_a_esterno
        if self.verbose:
            print("  Environment.modifica_interno: adatta a esterno = {}".format(adatta_a_esterno))

        if self.termosifone.acceso:
            fattore_adattamento_termosifone = 30
            adatta_a_termosifone = (self.termosifone.temp_propria - self.interno.temp)/fattore_adattamento_termosifone
            nuova_temp_interno += adatta_a_termosifone
            if self.verbose:
                print("  Environment.modifica_interno: adatta a termosifone = {}".format(adatta_a_termosifone))

        nuova_temp_interno = max( min (nuova_temp_interno,30),15)
        if self.verbose:
            print("  Environment.modifica_interno: nuova temp interno = {}".format(nuova_temp_interno))

        self.interno.modifica_temp(nuova_temp_interno)

    def arrotonda_mezza_unita(self,temp):
        temp *= 2
        temp = round(temp,0)
        temp /=2
        return temp

    def prossimo_stato(self,azione):

        self.modifica_termosifone(azione[0], azione[1], azione[2])
        self.modifica_interno()
        self.modifica_esterno()

    def calcola_stato(self):
        # esterno
        stato_esterno = self.arrotonda_mezza_unita(self.esterno.temp)
        #interno
        stato_interno = self.arrotonda_mezza_unita(self.interno.temp)

        if self.verbose:
            self.stato_seq.append((stato_esterno, stato_interno))
            print('  Environment.calcola_stato: stato_esterno, stato_interno = {} {}'.format(stato_esterno, stato_interno))
        return stato_esterno, stato_interno

    def modifica_termosifone(self,su,giu,centro):
        self.termosifone.set_points(su,centro,giu)
        self.verifica_accendi()

    def verifica_accendi(self):
        self.termosifone.termostato()

    def modifica_esterno(self):
        temp = 15
        #temp = max( min(self.esterno.temp + float(np.random.rand(1) * 2 - 1) , 40), 0)
        self.esterno.modifica_temp(temp)
