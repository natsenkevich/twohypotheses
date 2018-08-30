import numpy as np
import random as random
from scipy.stats import norm
import matplotlib.pyplot as plt

random.seed(4)
mu1=random.uniform(0, 1)
sigma1=random.uniform(0, 0.5)
mu2=random.uniform(0, 1)
sigma2=random.uniform(0, 0.5)

n=100
x1 = np.linspace(mu1 - 4*sigma1, mu1 + 4*sigma1, n)
norm1=norm.pdf(x1, mu1, sigma1)
plt.plot(x1, norm1)
plt.text(0.21,5,"$H_{0}$", size = 20)

x2 = np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, n)
norm2=norm.pdf(x2, mu2, sigma2)
plt.plot(x2,norm2) 
plt.text(0.38,3,"$H_{1}$", size = 20)

s1=np.zeros(n)
s2=np.zeros(n)
c= np.linspace(mu1 - 4*sigma1, mu1 + 4*sigma1, n)

for i in range(n):
    if c[i]<mu1:        
        xx1 = np.linspace(mu1 - 4*sigma1, c[i], n)
        xx2 = np.linspace(mu2 - 4*sigma2, mu1-(c[i]-mu1), n)
        s1[i]=2*np.trapz(norm1,xx1)
        s2[i]=np.trapz(norm2,xx2)
    if c[i]>=mu1:
        k=mu1-(c[i]-mu1)     
        xx1 = np.linspace(mu1 - 4*sigma1, k, n)
        xx2 = np.linspace(mu2 - 4*sigma2, c[i], n)
        s1[i]=2*np.trapz(norm1,xx1)
        s2[i]=np.trapz(norm2,xx2)

plt.fill_between(x2, norm2, 0, where=x2 < c[70], color='cyan')
plt.text(0.32,0.5,"$\\alpha$", size = 15)
plt.fill_between(x1, norm1, 0, where=x1 >=c[70], color='pink')
plt.fill_between(x1, norm1, 0, where=x1 < mu1-(c[70]-mu1), color='pink')
plt.text(0.25,0.5,"$\\beta$", size = 15)
plt.plot([c[70], c[70]], [0, 5])
plt.text(c[70], 5, 'c', size = 15)

plt.figure()
plt.plot(c,s1)
plt.plot(c,s2)
plt.title('Зависимость ошибки от критерия')