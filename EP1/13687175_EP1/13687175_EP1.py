import numpy as np

T = 1 # Number of experiments

def main():
    '''
    Testes da classe Area
    '''
    raio = 1
    print("---------- EXPERIMENTOS ----------\n")
    print('{0:^8} | {1:^10} | {2:^25} | {3:^15}'.format('n', 'Area (pi^)', 'pi^ - pi', 'Precisão (%)'))
    print('{0:-^70}'.format(''))

    for i in range(0, 20):
        np.random.seed(13687175 + i)
        area = Area(n=15000000, t=T, r=raio)
        print('{0:^8} | {1:^10} | {2:^25} | {3:^15}'.format(f'{area.n}',
                                                            f'{area.mean():.6}',
                                                            f'{area.erro_estimado()}',
                                                            f'{area.erro_relativo():.6}'))
    
    print()


class Area:

    def __init__(self, n, t = T, r = 1):
        self.n = n
        self.t = t
        self.raio = r

        area_circulo = 0
        area_circulo += self.experimento()
        
        self.p = 4*area_circulo/t

    def erro_estimado(self):
        return f'{abs(np.pi - self.mean())}'

    def erro_relativo(self):
        porc = abs((self.mean() / np.pi))*100
        if porc > 100:
            return 100 - (porc - 100)
        return porc

    
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

        xPoints = np.random.random(n) * 2 - 1  # create randomly points to the x axis
        yPoints = np.random.random(n) * 2 - 1 # create randomly points to the y axis

        # function that will tell us if each point is inside the circle
        indicadora = xPoints**2 + yPoints**2 < self.raio**2
        cont = np.sum(indicadora) # count the number of points that are inside the

        return r*r*(cont/n)  # probability that the point is inside the semicircle

if __name__ == '__main__':
    main()

