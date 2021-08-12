

coeffs = {
        'a':{ 1:85.334407, # a1 = 85.334407
                2:80.51249,  # a2 = 80.51249
                3:9.300961,  # a3 = 9.300961
                },
        'b':{ 1:0.0056858, # b1 = 0.0056858
                2:0.0071317, # b2 = 0.0071317
                3:0.0047026, # b3 = 0.0047026
                },
        'c':{ 1:0.0006262, # c1 = 0.0006262
                2:0.0029955, # c2 = 0.0029955
                3:0.0012547, # c3 = 0.0012547
                },
        'd':{ 1:0.0022053, # d1 = 0.0022053
                2:0.0021813, # d2 = 0.0021813
                3:0.0019085, # d3 = 0.0019085
                },
}

restrictions = {
                'min':{ 
                        1:78,
                        2:33,
                        3:27,
                        4:27,
                        5:27,
                        },
                'max':{ 
                        1:102,
                        2:45,
                        3:45,
                        4:45,
                        5:45,
                        },
}


def eq_i(x, coeffs, i):
    if i==1:
        eq = coeffs['a'][i] + coeffs['b'][i]*x[2]*x[5] + coeffs['c'][i]*x[1]*x[4] - coeffs['d'][i]*x[3]*x[5]  # g1, g2
    elif i==2:
        eq = coeffs['a'][i] + coeffs['b'][i]*x[2]*x[5] + coeffs['c'][i]*x[1]*x[2] + coeffs['d'][i]*x[3]*x[3]  # g3, g4
    elif i==3:  
        eq = coeffs['a'][i] + coeffs['b'][i]*x[3]*x[5] + coeffs['c'][i]*x[1]*x[3] + coeffs['d'][i]*x[3]*x[4]  # g5, g6
    else:
        raise ValueError('Choose from equation [1,2 or 3]')

    return eq

def fx(x):
    return 5.3578547*x[3]*x[3] + 0.8356891*x[1]*x[5] + 37.293239*x[1] - 40792.141  # f(x)

def check_restrictions(eq):
    if (eq[1] >= 0  and eq[1] <= 92  and 
        eq[2] >= 90 and eq[2] <= 110 and 
        eq[3] >= 20 and eq[3] <= 25 ):
        
        return True

    return False

def get_min_max(x_aux, restrictions, i, vizinhanca):

    maximo = restrictions['max'][i] if(x_aux[i] + vizinhanca > restrictions['max'][i]) else x_aux[i] + vizinhanca
    minimo = restrictions['min'][i] if(x_aux[i] - vizinhanca < restrictions['min'][i]) else x_aux[i] - vizinhanca
    
    return minimo, maximo

