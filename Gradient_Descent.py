import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

D=10

def Generate_C():
    C = np.random.normal(0,1, size=(D,1))
    return C

def Generate_Matrix():
    matrix = np.random.normal(0,1, size=(D,D))
    return matrix 

def Gradient_descent(x_start):
    x_old=x_start
    error=np.linalg.norm(np.subtract(x_old,x_star)) 
    count=0
    alpha=0.03
    loss=[]
    angle_list=[]
    while(error>0.01):
        diff=np.subtract(np.matmul(Q,x_old),C)                                         # Differential with respect to x.
        x_new=np.subtract(x_old,(np.dot(alpha,diff)))
        transpose_val=np.subtract(x_new,x_old)
        transpose_val=transpose_val.transpose()
        numerator=np.matmul(transpose_val,np.subtract(x_star,x_old))
        denominator=(np.linalg.norm(np.subtract(x_new,x_old)))*(np.linalg.norm(np.subtract(x_star,x_old)))
        angle=numerator/denominator                                                     # Calculating the angle between (X(k+1)-X(k)) and X*-X(k))
        for i in angle:                                                                 # Converting a list of list into a value
            b=i
        for i in b:
            angle=i    
        angle_list.append(angle)
        error=np.linalg.norm(np.subtract(x_new,x_star))                                 # Calculating the error from the true minimum
        loss.append(error)
        x_old=x_new
        count=count+1
        
    plt.plot(loss, label="LOSS Alpha")                                                  # Plotting the error curve
    plt.title("Constant Alpha")
    plt.show()
    plt1.plot(angle_list, label="ANGLE")                                                # Plotting the Angle
    plt1.title("Angle Constant Alpha")
    plt1.show()
    return count

def Gradient_descent_alpha(x_start):
    x_old=x_start
    #print("x_start ",x_old)
    error=np.linalg.norm(np.subtract(x_old,x_star)) 
    count_o=0
    loss=[]
    angle_list=[]
    while(error>0.01):
        p_k=np.subtract(np.matmul(Q,x_old),C)                                           # Differential with respect to x.
        p_k_t=p_k.transpose()
        numerator=np.matmul(p_k_t,p_k)
        denominator_1=np.matmul(p_k_t,Q)
        denominator=np.matmul(denominator_1,p_k)
        a=numerator/denominator                                                         # Calculating the optimal alpha value at every step
        for i in a:
            b=i
        for i in b:
            alpha=i    
        #print("alpha ",alpha)
        x_new=np.subtract(x_old,(np.dot(alpha,p_k)))                                    # Calculating the error from the true minimum
        #Gradient Angle
        transpose_val=np.subtract(x_new,x_old)
        transpose_val=transpose_val.transpose()
        n=np.matmul(transpose_val,np.subtract(x_star,x_old))
        d=(np.linalg.norm(np.subtract(x_new,x_old)))*(np.linalg.norm(np.subtract(x_star,x_old)))
        angle=n/d                                                                       # Calculating the angle between (X(k+1)-X(k)) and X*-X(k))
        #print(angle)
        for i in angle:
            b=i
        for i in b:
            angle=i    
        angle_list.append(angle)
        #Gradient Angle
        error=np.linalg.norm(np.subtract(x_new,x_star)) 
        loss.append(error)
        #print("error ",error)
        x_old=x_new
        count_o=count_o+1
    plt.plot(loss, label="LOSS Optimal Alpha")
    plt.title("Optimal Alpha")
    plt.show()
    plt1.plot(angle_list, label="ANGLE")
    plt1.title("Angle Optimal Alpha")
    plt1.show()
    return count_o

def Gradient_descent_Momentum(x_start):
    x_old=x_start
    error=np.linalg.norm(np.subtract(x_old,x_star)) 
    count_o=0
    alpha=0.03
    beta=0.8
    loss=[]
    q_k=0
    angle_list=[]
    while(error>0.01):
        p_k=np.subtract(np.matmul(Q,x_old),C)                                               # Calculation of the differential of F(x)
        #print("p_k ",p_k)
        if(count_o!=0):
            q_k=np.subtract(x_old,x_prev)                                                   # Momentum which will pull the iterate in the direction taken previously
        x_new=np.subtract(x_old,(np.dot(alpha,p_k)))
        x_new=np.add(x_new,(np.dot(beta,q_k)))
        error=np.linalg.norm(np.subtract(x_new,x_star))                                     # Error
        loss.append(error)  
        #print("error ",error)
        #Gradient Angle
        transpose_val=np.subtract(x_new,x_old)
        transpose_val=transpose_val.transpose()
        n=np.matmul(transpose_val,np.subtract(x_star,x_old))
        d=(np.linalg.norm(np.subtract(x_new,x_old)))*(np.linalg.norm(np.subtract(x_star,x_old)))
        angle=n/d                                                                            # Angle 
        #print(angle)
        for i in angle:
            b=i
        for i in b:
            angle=i    
        angle_list.append(angle)
        #Gradient Angle
        x_prev=x_old
        x_old=x_new
        count_o=count_o+1
    plt.plot(loss, label="LOSS Random alpha beta ")                                          # Plotting the error
    plt.show()
    plt.title("Constant Alpha Beta")
    plt1.plot(angle_list, label="ANGLE")                                                     # Plotting the angle
    plt1.title("Angle Constant Alpha Beta")
    plt1.show()
    return count_o

def calcAlpha(p_k,q_k):
    numerator_1=np.matmul(q_k.transpose(),p_k)
    numerator_1=np.matmul(numerator_1,p_k.transpose())
    numerator_1=np.matmul(numerator_1,Q)
    numerator_1=np.matmul(numerator_1,q_k)

    numerator_2=np.matmul(p_k.transpose(),p_k)
    numerator_2=np.matmul(numerator_2,q_k.transpose())
    numerator_2=np.matmul(numerator_2,Q)
    numerator_2=np.matmul(numerator_2,q_k)

    numerator=np.subtract(numerator_1,numerator_2)

    denominator_1=np.matmul(p_k.transpose(),Q)
    denominator_1=np.matmul(denominator_1,q_k)
    denominator_1=np.matmul(denominator_1,p_k.transpose())
    denominator_1=np.matmul(denominator_1,Q)
    denominator_1=np.matmul(denominator_1,q_k)

    denominator_2=np.matmul(p_k.transpose(),Q)
    denominator_2=np.matmul(denominator_2,p_k)
    denominator_2=np.matmul(denominator_2,q_k.transpose())
    denominator_2=np.matmul(denominator_2,Q)
    denominator_2=np.matmul(denominator_2,q_k)

    denominator=np.subtract(denominator_1,denominator_2)

    a=numerator/denominator
    for i in a:
        b=i
    for i in b:
        alpha=i  
    return alpha  

def calcBeta(p_k,q_k):
    numerator_1=np.matmul(p_k.transpose(),p_k)
    numerator_1=np.matmul(numerator_1,p_k.transpose())
    numerator_1=np.matmul(numerator_1,Q)
    numerator_1=np.matmul(numerator_1,q_k)

    numerator_2=np.matmul(q_k.transpose(),p_k)
    numerator_2=np.matmul(numerator_2,p_k.transpose())
    numerator_2=np.matmul(numerator_2,Q)
    numerator_2=np.matmul(numerator_2,p_k)
    numerator=np.subtract(numerator_1,numerator_2)

    denominator_1=np.matmul(q_k.transpose(),Q)
    denominator_1=np.matmul(denominator_1,q_k)
    denominator_1=np.matmul(denominator_1,p_k.transpose())
    denominator_1=np.matmul(denominator_1,Q)
    denominator_1=np.matmul(denominator_1,p_k)

    denominator_2=np.matmul(p_k.transpose(),Q)
    denominator_2=np.matmul(denominator_2,q_k)
    denominator_2=np.matmul(denominator_2,p_k.transpose())
    denominator_2=np.matmul(denominator_2,Q)
    denominator_2=np.matmul(denominator_2,q_k)

    denominator=np.subtract(denominator_1,denominator_2)

    a=numerator/denominator
    for i in a:
        b=i
    for i in b:
        beta=i  

    return beta       


    numerator_1=np.matmul(p_k.transpose(),p_k)
    numerator_1=np.matmul(numerator_1,p_k.transpose())
    numerator_1=np.matmul(numerator_1,Q)
    numerator_1=np.matmul(numerator_1,q_k)

    numerator_2=np.matmul(q_k.transpose(),p_k)
    numerator_2=np.matmul(numerator_2,p_k.transpose())
    numerator_2=np.matmul(numerator_2,Q)
    numerator_2=np.matmul(numerator_2,p_k)
    numerator=np.subtract(numerator_1,numerator_2)

    denominator_1=np.matmul(q_k.transpose(),Q)
    denominator_1=np.matmul(denominator_1,q_k)
    denominator_1=np.matmul(denominator_1,p_k.transpose())
    denominator_1=np.matmul(denominator_1,Q)
    denominator_1=np.matmul(denominator_1,p_k)

    denominator_2=np.matmul(p_k.transpose(),Q)
    denominator_2=np.matmul(denominator_2,q_k)
    denominator_2=np.matmul(denominator_2,p_k.transpose())
    denominator_2=np.matmul(denominator_2,Q)
    denominator_2=np.matmul(denominator_2,q_k)

    denominator=np.subtract(denominator_1,denominator_2)

    a=numerator/denominator
    for i in a:
        b=i
    for i in b:
        beta=i  

    return beta  

def Gradient_descent_Momentum_alpha(x_start):
    x_old=x_start
    error=np.linalg.norm(np.subtract(x_old,x_star)) 
    count_o=0
    loss=[]
    angle_list=[]
    q_k=0
    while(error>0.01):
        p_k=np.subtract(np.matmul(Q,x_old),C)
        if(count_o!=0):
            q_k=np.subtract(x_old,x_prev)
            alpha=calcAlpha(p_k,q_k)
            beta=calcBeta(p_k,q_k)
        else:
            alpha=0.03  
            beta=0    
        x_new=np.subtract(x_old,(np.dot(alpha,p_k)))
        x_new=np.add(x_new,(np.dot(beta,q_k)))
        error=np.linalg.norm(np.subtract(x_new,x_star)) 
        loss.append(error)
        transpose_val=np.subtract(x_new,x_old)
        transpose_val=transpose_val.transpose()
        n=np.matmul(transpose_val,np.subtract(x_star,x_old))
        d=(np.linalg.norm(np.subtract(x_new,x_old)))*(np.linalg.norm(np.subtract(x_star,x_old)))
        angle=n/d
        for i in angle:
            b=i
        for i in b:
            angle=i    
        angle_list.append(angle)
        x_prev=x_old
        x_old=x_new
        count_o=count_o+1
    plt.plot(loss, label="LOSS Random alpha beta ")
    plt.title("Optima Alpha Beta")
    plt.show()
    plt1.plot(angle_list, label="ANGLE")
    plt1.title("Angle Optimal Alpha Beta")
    plt1.show()
    return count_o

def calcOrthogonal(p_k):
    q_k = np.array(np.random.choice([0], size=(D,1)),dtype = float) 
    q_k[8][0]=p_k[8][0]
    q_k[9][0]= - ((p_k[8][0]*p_k[8][0]) / p_k[9][0])
    print("dot_prod ",np.dot(p_k.transpose(),q_k))
    return q_k
     
def Gradient_Descent_Orthogonal(x_start):
    x_old=x_start
    error=np.linalg.norm(np.subtract(x_old,x_star)) 
    count_o=0
    loss=[]
    angle_list=[]
    q_k=0
    while(error>0.01):
        p_k=np.subtract(np.matmul(Q,x_old),C)
        q_k=calcOrthogonal(p_k)
        alpha=calcAlpha(p_k,q_k)
        beta=calcBeta(p_k,q_k)
        x_new=np.subtract(x_old,(np.dot(alpha,p_k)))
        x_new=np.add(x_new,(np.dot(beta,q_k)))
        error=np.linalg.norm(np.subtract(x_new,x_star)) 
        loss.append(error)
        print("error ",error)
        transpose_val=np.subtract(x_new,x_old)
        transpose_val=transpose_val.transpose()
        n=np.matmul(transpose_val,np.subtract(x_star,x_old))
        d=(np.linalg.norm(np.subtract(x_new,x_old)))*(np.linalg.norm(np.subtract(x_star,x_old)))
        angle=n/d
        for i in angle:
            b=i
        for i in b:
            angle=i    
        angle_list.append(angle)
        x_prev=x_old
        x_old=x_new
        count_o=count_o+1
    plt.plot(loss, label="LOSS Random alpha beta ")
    plt.title("Orthogonal Descent")
    plt.show()
    plt1.plot(angle_list, label="ANGLE")
    plt1.title("Orthogonal Descent Angle")
    plt1.show()
    return count_o

x_start=np.random.normal(0,1, size=(D,1))
constant_alpha_list=[]
optimal_alpha_list=[]
constant_alpha_beta_list=[]
optimal_alpha_beta_list=[]
orthogonal_alpha_beta_list=[]
for i in range(1):
    print(i)
    C=Generate_C()
    A=Generate_Matrix()  
    A_t=A.transpose()
    Q=np.matmul(A,A_t)
    Q_inverse=np.linalg.inv(Q)
    x_star=np.matmul(Q_inverse,C)
    #print(Q)    
    constant_alpha = Gradient_descent(x_start)
    optimal_alpha = Gradient_descent_alpha(x_start)
    constant_alpha_beta = Gradient_descent_Momentum(x_start)
    optimal_alpha_beta=Gradient_descent_Momentum_alpha(x_start)
    orthogonal_alpha_beta=Gradient_Descent_Orthogonal(x_start)
    print("constant_alpha ",constant_alpha)
    print("optimal_alpha  ",optimal_alpha)
    print("constant_alpha_beta ",constant_alpha_beta)
    print("optimal_alpha_beta ",optimal_alpha_beta)
    print("orthogonal_alpha_beta ",orthogonal_alpha_beta)
    constant_alpha_list.append(constant_alpha)
    optimal_alpha_list.append(optimal_alpha)
    constant_alpha_beta_list.append(constant_alpha_beta)
    optimal_alpha_beta_list.append(optimal_alpha_beta)
    orthogonal_alpha_beta_list.append(orthogonal_alpha_beta)
print("constant_alpha_list ",constant_alpha_list)
print("optimal_alpha_list ",optimal_alpha_list)    
print("constant_alpha_beta_list ",constant_alpha_beta_list)
print("optimal_alpha_beta_list ",optimal_alpha_beta_list)    
print("orthogonal_alpha_beta_list ",orthogonal_alpha_beta_list)    
"""
C=Generate_C()
A=Generate_Matrix()  
A_t=A.transpose()
Q=np.matmul(A,A_t)
Q_inverse=np.linalg.inv(Q)
x_star=np.matmul(Q_inverse,C)
x_start=np.random.normal(0,1, size=(D,1))
constant_alpha = Gradient_descent(x_start)
optimal_alpha = Gradient_descent_alpha(x_start)
constant_alpha_beta = Gradient_descent_Momentum(x_start)
optimal_alpha_beta=Gradient_descent_Momentum_alpha(x_start)
orthogonal_alpha_beta=Gradient_Descent_Orthogonal(x_start)
print("constant_alpha ",constant_alpha)
print("optimal_alpha  ",optimal_alpha)
print("constant_alpha_beta ",constant_alpha_beta)
print("optimal_alpha_beta ",optimal_alpha_beta)
print("orthogonal_alpha_beta ",orthogonal_alpha_beta)

"""

