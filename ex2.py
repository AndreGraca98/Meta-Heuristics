# %% 
import numpy as np
import matplotlib.pyplot as plt
import os, datetime, argparse



def run(args):
    # Parâmetros
    if args.exercise==1:
        Tmax =  100 # 80 10
        Tmin = 0.1
        neighboor_base = .5  # .5 1
        iterations = 10

        graph = np.zeros((1,1000000))
        graph_idx = 0
        T = Tmax
        t = 1
    
        cool_cooeff = args.coefficients #  # coeeficiente de arrefecimento
        # cool_cooeff = 1e-5  # coeeficiente de arrefecimento
    
    elif args.exercise==2:
        Tmax =  10 # 80 10
        Tmin = 0.1
        neighboor_base = 2  # .5 1
        iterations = 10

        graph = np.zeros((1,1000000))
        graph_idx = 0
        T = Tmax
        t = 1

        cool_cooeff = args.coefficients  # coeeficiente de arrefecimento
        # cool_cooeff = 1e-4  # coeeficiente de arrefecimento

    x = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_best = {1:0, 2:0, 3:0, 4:0, 5:0}
    x_aux = {1:0, 2:0, 3:0, 4:0, 5:0}
    eq = {1:0, 2:0, 3:0}


    rn = np.random.random  # random number between {0, 1}

    # Passo 1, dar valores a x, gi(x), f(x)
    flag = False
    while not flag:
        for i in range(1,6):  # 1 - 5
            x[i] = restrictions['min'][i] + rn() * (restrictions['max'][i]-restrictions['min'][i])  

        for i in range(1,4):  # gi(x)  1 - 3
            eq[i] = eq_i(x, coeffs, i)
            
        # Se as restrições forem verificadas, o algoritmo pode seguir
        flag = check_restrictions(eq)

    best_result = fx(x)  # f(x)


    # Best results
    best = best_result
    for i in range(1,6):
        x_best[i] = x[i]



    # Visualizing data
    graph[0,graph_idx] = best_result
    graph_idx += 1



    while T > Tmin:
        
        iteration = 0
        
        #  Passo 3, seleciona um ponto na vizinhança de best_result
        while iteration < iterations:
            
            neighboor = neighboor_base
            for i in range(1,6):
                x_aux[i] = x[i]
            
            # valores para entrar nos ciclos
            flag = False
            counter, counter_max = 0, 1000
            while not flag and counter < counter_max:
                
                # Define new x
                for i in range(1,6):
                    minimo, maximo = get_min_max(x_aux, restrictions, i, neighboor)
                    x[i] = minimo + rn()*(maximo - minimo)
                
                for i in range(1,4):  # gi(x)
                    eq[i] = eq_i(x, coeffs, i)
        
        
                flag = check_restrictions(eq)

                counter += 1
                
                # Depois de um certo número de iterações é conveniente reduzir o tamanho da pesquisa na vizinhança, caso contrário podemos ficar em ciclos infinitos
                if (counter == round(counter_max/5) or
                    counter == round(2*counter_max/5) or
                    counter == round(3*counter_max/5) or
                    counter == round(4*counter_max/5)):

                    neighboor -= .24 * neighboor_base
                
            
            # Passo 4, avaliação da nova soluçao
            # Se o resultado anterior tiver sido não admissível, termina o programa
            if not flag:
                break
            
            
            result = fx(x)   # result = 5.3578547*x[3]*x[3] + 0.8356891*x[1]*x[5] + 37.293239*x[1] - 40792.141
            

            if result < best_result:  # minimizar
                # Se for melhor, substitui o ponto anterior
                best_result = result
                graph[0,graph_idx] = best_result
                graph_idx += 1

                if best_result < best:
                    best = best_result
                    for i in range(1,6):
                        x_best[i] = x[i]
                
            elif rn() < np.exp((best_result - result)/T):
                # Caso contrário, existe uma hipótese de substituí-lo na mesma,
                # baseado no quão pior é esse ponto
                best_result = result
                graph[0,graph_idx] = best_result
                graph_idx += 1
            
            iteration += 1
        
        
        # Passo 5, "Arrefecer" o problema e voltar ao passo 3 usando o coeficiente de arrefecimento  alpha
        # Se o resultado anterior tiver sido não admissível, termina o programa!
        alpha = np.exp(-cool_cooeff*t);   # coeficiente de arrefecimento para reduzir T
        if not flag:
            break
        
        
        T = Tmax * alpha
        t += 1

        print(f'\rT:{T:.3f}\t\tt:{t:.2e}\tf(x): {best_result:.3f}', end='')


    # print(x_best)
    # print(best_result)




    # Filtered graph . Remove zeros
    g = graph[0][:(graph[0]==0).argmax()]



    #Save data
    if not os.path.isdir('data'):
        os.makedirs('data')

    filename = f'data/SA_{args.exercise}_{args.coefficients:.0e}'
    # filename = f'data/SA_{datetime.datetime.now().strftime("%d%b_%H_%M")}_{args.exercise}_{args.coefficients:.0e}'

    plt.plot(g , linewidth=1)
    plt.savefig(filename + '.png')
    plt.show()

    with open(filename + '.txt','w') as f:
        variables = {'Tmax':Tmax,
                     'Tmin':Tmin,
                     'neighboor_base':neighboor_base,
                     'iterations':iterations,
                     'cool_cooeff':cool_cooeff,}
        variables = '\n'.join([f'{k}: {v:.5f}' for k, v in variables.items() ])
        xx = '\n'.join([f'x{k}: {v:.5f}' for k, v in x_best.items() ])

        f.write(f"{filename[8:]}\nEx2.{args.exercise}\n\n{xx}\nf(x): {best_result}\n\n\n{variables}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ex2')
    parser.add_argument('-e', '--exercise', type=int, choices={1,2},help='Exercise 2.1 [1] or 2.2 [2]')
    parser.add_argument('-c', '--coefficients', type=float, choices={1e-5, 1e-4, 1e-3, 1e-2}, default=1e-3 ,help='Cooling coefficients')
    args = parser.parse_args()


    if args.exercise==1:
        from ex21 import * 
    elif args.exercise==2:
        from ex22 import * 
    else:
        raise ValueError('Choose exercise 2.1 [1] or 2.2 [2]')

    run(args)
    print()


