from math import exp,sin,cos
import time
import socket

def trapz(F,a,b,n):
    h=(b-a)/n
    sum=0.5*(F(a)+F(b))
    for i in range(1,n):
        sum+=F(i*h)
    return sum*h

def f(x):
    return exp(-x)*x*x

def g(x):
    if x<0.5:
        h=-exp(-x)
    else:
        h= exp(x)
    return h*x*x

def implicit(t):
    # implicit = root of  4*sin(x)-exp(x)+t
    # Newton iterations, starting from zero:    
    x=0.0
    F= 4*sin(x)-exp(x)+t
    while abs(F)> 1.e-15:
        x-= F/(4*cos(x) - exp(x))
        F=  4*sin(x)-exp(x)+t
    return x

#----------------------main program starts here ------------------
loops=10000
n=1000

fic=open("RunningOn"+socket.gethostname(),"w")

for F in [f,g,implicit]:
    t1 = time.time()
    for i in range(0,loops):
        sum=trapz(F,0.0,1.0,n)
    t=(time.time()-t1)/loops
    print(F.__name__," ",t," ",sum)
    fic.write(F.__name__+": "+str(t)+"\n")
fic.close()
print("end.")
