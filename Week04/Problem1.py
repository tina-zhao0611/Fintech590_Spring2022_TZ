import numpy as np

#generating rt series
n = 100000
sigma2 = 2
rt = np.sqrt(sigma2) * np.random.randn(n - 1)

#calculating Pt
Pt = np.zeros(n) 
Pt_1 = 1

print("rt ~ N( 0,", sigma2,")\nPt-1 = ", Pt_1)

#Classical Brownian Method
for i in range(1, n):
    Pt[i] = Pt_1 + rt[i - 1]
    #print result along with calculated expected values
print("Classical Brownian Method: \n E(P) = ", np.mean(Pt), "Expected(Pt-1):", Pt_1)
print("Var(P) = ", np.var(Pt), "Expected(σ^2):", sigma2)

#Arithmetic Return
for i in range(1, n):
    Pt[i] = Pt_1 * (1 + rt[i - 1])
    #print result along with calculated expected values
print("Arithmetic Return: \n E(P) = ", np.mean(Pt), "Expected(Pt-1):", Pt_1)
print("Var(P) = ", np.var(Pt), "Expected(Pt-1^2*σ^2):", Pt_1**2 * sigma2)

#Geometic Brownian Motion
for i in range(1, n):
    Pt[i] = Pt_1 * np.exp(rt[i - 1])
    #print result along with calculated expected values
print("Geometic Brownian Motion: \n E(P) = ", np.mean(Pt), "Expected(e^(σ^2/2) * Pt-1):", np.exp(sigma2/2)* Pt_1)
print("Var(P) = ", np.var(Pt), "Expected(Pt-1^2 * (e^(σ^2)-1) ) * e^(σ^2)):", Pt_1**2 * (np.exp(sigma2) - 1)* np.exp(sigma2))
