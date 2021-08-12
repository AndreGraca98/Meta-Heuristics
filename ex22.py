import numpy as np

coeffs = {}

restrictions = {
                'min':{ 
                        1:-2.3,
                        2:-2.3,
                        3:-3.2,
                        4:-3.2,
                        5:-3.2,
                        },
                'max':{ 
                        1:2.3,
                        2:2.3,
                        3:3.2,
                        4:3.2,
                        5:3.2,
                        },
}


def eq_i(x, coeffs, i):
    if i==1:
        eq = x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + x[4]*x[4] + x[5]*x[5]  # g1
    elif i==2:
        eq = x[2]*x[3] - 5 * x[4]*x[5] # g2
    elif i==3:  
        eq = x[1]*x[1]*x[1] + x[2]*x[2]*x[2] # g3
    else:
        raise ValueError('Choose from equation [1,2 or 3]')

    return eq

def fx(x):
    return np.exp(x[1]*x[2]*x[3]*x[4]*x[5])  
    
def check_restrictions(eq, xtra = .25):
    if (eq[1] <= 10+xtra and eq[2] <= 0+xtra and eq[3] <= 1+xtra and
        eq[1] >= 10-xtra and eq[2] >= 0-xtra and eq[3] >= 1-xtra) :
        return True
    return False

def get_min_max(x_aux, restrictions, i, vizinhanca):

    maximo = restrictions['max'][i] if(x_aux[i] + vizinhanca > restrictions['max'][i]) else x_aux[i] + vizinhanca
    minimo = restrictions['min'][i] if(x_aux[i] - vizinhanca < restrictions['min'][i]) else x_aux[i] - vizinhanca
    
    return minimo, maximo

