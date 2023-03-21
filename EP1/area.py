import random
import matplotlib.pyplot as plt
import numpy as np

T = 32 # Number of experiments

def main():
    '''
    Testes da classe Area
    '''

    print("---------- EXPERIMENTOS DA CLASSE AREA ----------")
    print(f"Nº de experimentos utilizados no teste = {T}\n")
    print('{0:^5} | {1:^30} | {2:^30} | {3:^10}'.format('n',
                                                        'Area',
                                                        'Erro estimado',
                                                        '% do ERRO'))
    print('{0:-^80}'.format(''))

    raio = 1
    for i in range(20):
        area = Area(n = 2**i, t=T, r=raio)
        print('{0:^5} | {1:^30} | {2:^30} | {3:^10}'.format(f'{area.n}',
                                                            f'{area.mean()}',
                                                            f'{area.erro_estimado()}',
                                                            f'{area.erro_relativo()}'))
    
    print()


class Area:

    def __init__(self, n, t = T, r = 1):
        self.n = n
        self.t = t
        self.raio = r
        xPoints = np.random.random(n)
        yPoints = np.random.random(n)

        area_semicirculo = 0
        for i in range(t):
            area_semicirculo += self.experimento()
        
        #self.geraGrafico(self.points)
        
        self.p = 4*area_semicirculo/t

    def erro_estimado(self):
        return f'{np.pi - self.mean()}'

    def erro_relativo(self):
        return f'{self.mean() / np.pi}'
    
    def mean(self):
        '''(Area) -> float
        RETORNA a estimativa experimental da área de um círculo de
           raio self.r
        '''
        return self.p

    def experimento(self):
        '''(Area) -> float
        SIMULA um experimento sorteando self.n pontos no
            quadrante de lado self.r .
        RETORNA um estimativa da área do circulo de raio self.r
        '''

        n = self.n
        r = self.raio
        if n == 0: return 0  # there's no experiment to do

        self.xPoints = np.random.random(n)  # create randomly points to the x axis
        self.yPoints = np.random.random(n)  # create randomly points to the y axis

        self.indicadora = self.xPoints**2 + self.yPoints**2 < self.raio**1  # function that will tell us if each point is inside the circle
        cont = np.sum(self.indicadora) # count the number of points that are inside the circle

        return r*r*(cont/n)  # probability that the point is inside the semicircle
    

    # ESSA FUNÇÃO FUNCIONA! MAS DEMORA MUITO PARA EXECUTAR
    def geraGrafico(self, points):
        an = np.linspace(0, 0.5 * np.pi, 50)  # with (0.5 * np.pi) we create a semicircle
        fig, axs = plt.subplots()
        
        axs.plot(np.cos(an), np.sin(an))
        axs.axis('equal')
        axs.set_title(f'Semicircle area (n = {self.n})', fontsize=10)

        for coordinate in points:
            x,y = coordinate
            if x*x + y*y < self.raio * self.raio:
                axs.scatter(x, y, color="black")
            else: axs.scatter(x, y, color="red")

        fig.tight_layout()

        plt.show()
    

if __name__ == '__main__':
    main()

