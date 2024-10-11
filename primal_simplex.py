'''
    Primal Symplex 2 fases Method
    MS428 - Unicamp
'''
# Importação da biblioteca que será utilizada
import numpy as np

# Definindo número de casas decimais pra efeito de visualização
np.set_printoptions(precision=3)

# Criação da função que implementa o método
def relativ_costs(not_base, base, c, basic_index, nonbasic_index):
    c_nb = c[nonbasic_index]
    Lambda = np.transpose(np.linalg.solve(np.transpose(base), c[basic_index]))#vetor multiplicador simplex
    c_relativ = np.zeros(c_nb.shape)

    for i in range(len(c_nb)):
        c_relativ[i] = c_nb[i] - Lambda@(not_base[:,i])
    entering_var_ind =  np.argmin(c_relativ)
    if np.min(c_relativ)>= 0: # condição de parada : solução ótima encontrada. Todos os custos são positivos.
        return True
    return  entering_var_ind

def primalsimplex_method(A, m, n, b, c, initial_solution=None, basic_index=None, max_iterations = 10000):
    base = np.zeros((m, m))

    # Inicialização com a solução básica inicial, se fornecida
    # função do numpy pra resolver sist linear linalg.solve(a, b)
    # função do numpy para transpor matriz: numpy.transpose(a, axes=None)

    if initial_solution is not None: 
        not_base = np.zeros((m,n-m)) #Nesse caso estamos na Fase II, então a particção não básica tem n-m colunas
        nonbasic_index =[]
        for i in range(n):
            if i not in basic_index:
                nonbasic_index.append(i)
        np.array(nonbasic_index)
    else: 
        not_base = np.zeros((m,n)) #Aqui estamos na Fase 1, então a matriz B da Base será a identidade mxm devido as var. artificiais
        basic_index = np.arange(n, n+m)
        nonbasic_index = np.arange(n)
    

    for iteration in range(max_iterations):
        base = A[:, basic_index]
        not_base = A[:, nonbasic_index]
        x_solution = np.linalg.solve(base,b)
        z = np.transpose(c[basic_index]) @ x_solution #valor da função atual
        
        entering_var_ind =  relativ_costs(not_base,base,c,basic_index,nonbasic_index)  # Variável que entra na base
        
        if type(entering_var_ind) == bool and np.min(x_solution) >= 0:
            # A solução é ótima
            x_optimal = np.zeros(n)
            count = 0
            for i in basic_index:
                x_optimal[i] = x_solution[count]
                count +=1
            
            return x_optimal, z, True, basic_index, iteration+1
        elif type(entering_var_ind) == bool and np.min(x_solution) < 0:
            return None, None, False, basic_index, iteration+1
        
        entering_var = nonbasic_index[entering_var_ind]

        # Escolher a variável que sai da base
        direction = np.linalg.solve(base, A[:,entering_var]) # cálculo da direção simplex utilizando a coluna com a nova variável básica
        leaving_var_ind = None #declaração da variável que sairá da base
        if np.max(direction) > 0:
            step = -1
            for i in range(len(direction)):
                y = direction[i]
                if y>0:
                    steps = abs((x_solution[i]/y))
                    if steps<step or step<0:
                        step = steps
                        leaving_var_ind = i #obtém o indice da variável com menor tamanho de passo             


        if leaving_var_ind is None:
            # O problema é ilimitado pois todas as entradas do vetor direção são menores que zero
            print("O problema é ilimitado.")
            return None, None, False, basic_index, iteration

        # Atualizar a base
        leaving_var = leaving_var_ind
        basic_index[leaving_var_ind] = entering_var #troca os indices, modificando a base B
        nonbasic_index[entering_var_ind] = leaving_var
    # Limite máximo de iterações atingido
    print("Limite de iterações.")
    return None, None, False, basic_index, iteration


def two_phase_simplex(A, b, c, m, n): 
    num_artificial_vars = m  # Iremos adicionar uma variável artificial para cada restrição

    # Fase 1: Encontrar uma solução viável básica inicial
    c_phase1 = np.concatenate((np.zeros(n), np.ones(num_artificial_vars)))  # Coeficientes da função objetivo da fase 1
    A_phase1 = np.hstack((A, np.eye(num_artificial_vars)))  # Matriz de coeficientes da fase 1. 
    #Temos uma submatriz que é igual a identidade referente aos coeficientes das var. artificiais.
    
    x_phase1, z_phase1, success_phase1, basic_ind, iteration1 = primalsimplex_method(A_phase1, m, n, b, c_phase1)

    if (success_phase1 is not True):
        print("Problema incompatível. Não há solução viável básica inicial.")
        return None

    # Fase 2: Otimizar a solução encontrada na fase 1
    x_phase2, z_phase2, success_phase2, basic_ind, iteration = primalsimplex_method(A, m, n, b, c, initial_solution=x_phase1, basic_index = basic_ind)

    if success_phase2 :
        return x_phase2, z_phase2, iteration + iteration1
    
    print('Não foi encontrado uma solução ótima.')
    return None

def main():
    
    # Leitura das entradas
    m = int(input("Número m de equações:"))
    n = int(input("Número n de variáveis:"))
    c = np.array([float(x) for x in input("Vetor c (custos): ").split()])
    b = np.array([float(x) for x in input("Vetor b (recursos): ").split()])
    A = []
    print('Linha por linha da Matriz A :')
    for i in range(m):
        row = np.array([float(x) for x in input().split()])
        A.append(row)
    A= np.array(A)
    
    print("======== SAÍDA: ========")
    success = two_phase_simplex(A, b, c, m, n) # a função retorna o vetor solução x e o valor z ótimo para a função f(x)
    if (success is not None):
        x, z, iteration = success
        print("Vetor das variáveis de decisão: " + str(x))
        print(f"Valor  ótimo da solução: {z:.4f}")
        print(f"Número de iterações: {iteration}")
    
main()
