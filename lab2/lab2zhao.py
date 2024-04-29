#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:26:27 2022
Modified on Fri Mar 11 15:26:27 2022
@author: Your name

Description
------------
"""

# Part 1 - Histogram the data

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as st

fig = plt.figure(1,figsize=(6,6))
fig.clf()
axes = [fig.add_subplot(321),\
        fig.add_subplot(322),\
        fig.add_subplot(323),\
        fig.add_subplot(324),\
        fig.add_subplot(325),\
        fig.add_subplot(326)]

# You can use axes[0], axes[1], ....  axes[5] to make the six histograms.

# Your code goes here
RefractData = np.loadtxt('refractionData.txt',skiprows=3)
for i in range(0,6):
    axes[i].hist(RefractData[i],bins=15,range=(-10,50))
    deg = 10*(i+1)
    axes[i].set_title(f'alpha = {deg:2.0f} deg.')
    axes[i].set_xlabel('beta (deg.)')


#%%

# Part 2 - Table of measurements

# Your code goes here

n_air = 1

Datarad = np.zeros((6,16))
StatInfo = np.zeros((6,2))
SinStats = np.zeros((6,2))

for i in range(0,6):
    for j in range(0,16):
        Datarad[i,j] = np.pi*RefractData[i,j]/180
    mean = Datarad[i].sum()/16
    StatInfo[i,0]=mean
    std = np.std(Datarad[i],ddof=1)
    std_mean = std/4
    StatInfo[i,1]=std_mean
    SinStats[i,1] = np.sin(StatInfo[i,1])
    SinStats[i,0] = np.sin(StatInfo[i,0])
    print(f'Alpha = {(10*(i+1)):2.0f} deg | sin(alpha) = {np.sin(10*(i+1)):2.2f} | Beta Mean = {StatInfo[i,0]:2.2f} | sin(beta_mean) = {SinStats[i,0]:2.2f} | stdev(sin(beta_mean)) | {SinStats[i,1]:2.2f}')







#%%

# Part 3 - Snells law plot and fit

fig = plt.figure(2,figsize=(6,6))
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# You can use ax1 and ax2 for the Snell's law plot and the chi squared plot.

# Your code goes here



SinB = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
SinA = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
UncertB = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
CheckA = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

for i in range(0,6):
    SinB[i] = SinStats[i,0]



for i in range(0,6):
    UncertB[i] = SinStats[i,1]

for i in range(0,6):
    SinA[i] = np.sin((10*(i+1))*np.pi/180)
    CheckA[i] = 10*(i+1)

ax1.errorbar(SinA, SinB, yerr = UncertB, fmt='ok')
ax1.set_xlabel(r"$sin(\alpha)$")
ax1.set_ylabel(r"$sin(\beta)$")

def Snell(x,a):
    return x/a

Par, Var = curve_fit(Snell, SinA, SinB, p0 = 1, sigma = UncertB)
ax1.plot(SinA, Snell(SinA, Par))
print(f"The predicted index of refraction is {Par[0]:2.2f} with an uncertainty of {np.sqrt(Var[0,0]):0.6f}")
chi1 = ( ((SinB-Snell(SinA,Par) )**2).sum() / (UncertB**2).sum())
degf1 = 6-1
p1 = st.chi2.sf(chi1, degf1)

print(f"The corresponding chi-squared to this fit is {chi1:2.2f} with a p-value of {p1:1.4f} with 5 degrees freedom")








# Part 4 - Chi squared plot

# Your code goes here

xx4 = np.linspace(1.4,1.55,201)
yy4 = np.zeros(201)



for i in range(len(yy4)):
    yy4[i] = ( ((SinB-Snell(SinA,xx4[i]) )**2) / (UncertB**2)).sum()
    
minimum = min(yy4)
maximum = max(yy4)

ax2.plot(xx4,yy4)
ax2.vlines(Par, 0,14)
ax2.vlines(Par+np.sqrt(Var[0,0]), 0,14, ls='--')
ax2.vlines(Par-np.sqrt(Var[0,0]), 0,14, ls='--')
ax2.hlines(minimum,1.4,1.55, ls='--')
ax2.hlines(( ((SinB-Snell(SinA,Par-np.sqrt(Var[0,0])) )**2) / (UncertB**2)).sum(),1.4,1.55, ls='--')

## The vertical and horizontal lines intersect correctly.

ax1.set_xlabel(r"$sin(\alpha)$")
ax1.set_ylabel(r"$sin(\beta)$")
ax1.set_title(r"$sin(\beta)$" ' v ' r"$sin(\alpha)$")

ax2.set_xlabel('Index of Refraction')
ax2.set_ylabel('$\chi^2$')
ax2.set_title('Index o. Refraction v $\chi^2$')

######################################################
