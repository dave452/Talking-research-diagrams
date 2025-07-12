# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:40:19 2023

@author: davem
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib as mpl

plt.style.use('default')
plt.rcParams.update({'xtick.labelsize' : 14,
                     'ytick.labelsize' : 14,
                     'axes.formatter.useoffset' : False,
                     'legend.fontsize' : 20,
                     'axes.labelsize' : 30,
                     "text.usetex": False,
                     'lines.linewidth':1,
                     'figure.figsize' : (8,8),
                     'figure.autolayout': True})

cols = mpl.colormaps['coolwarm']
print('Here')
#%% Confining
y1s = np.linspace(1,2,20)
y2s = np.linspace(-1,-2,20)
x1 = 0
x2 = 0
r_min = 1
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
i = 0
pos1 = [x1, y1s[i]]
pos2 = [x2, y2s[i]]
r = ((((pos1[0]-pos2[0])** 2) + ((pos1[1]-pos2[1])** 2))**0.5)/2
dy = 1/r**2
plt.arrow(pos1[0], pos1[1], 0, -dy, width = 0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') #
plt.arrow(pos2[0], pos2[1], 0, dy, width =0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') # ,length_includes_head=True
plt.text(.2, pos1[1] - dy / 2, 'F',fontsize=20)
plt.text(.2, pos2[1] + dy / 2, 'F',fontsize=20)
plt.xlim((-1.1,1.1))
plt.ylim((-2.5,2.5))
ax = plt.gca()
a = plt.Circle((pos1[0],pos1[1]),0.1, fc = 'w',ec='k')
ax.add_artist(a)
b = plt.Circle((pos2[0],pos2[1]),0.1, fc = 'w',ec='k')
ax.add_artist(b)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
i = -1
plt.subplot(1,2,2)
pos1 = [x1, y1s[i]]
pos2 = [x2, y2s[i]]
dy = 1/r**2
plt.arrow(pos1[0], pos1[1], 0, -dy, width = 0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') #
plt.arrow(pos2[0], pos2[1], 0, dy, width =0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') # ,length_includes_head=True
plt.text(.2, pos1[1] - dy / 2, 'F',fontsize=20)
plt.text(.2, pos2[1] + dy / 2, 'F',fontsize=20)
plt.xlim((-1.1,1.1))
plt.ylim((-2.5,2.5))
ax = plt.gca()
a = plt.Circle((pos1[0],pos1[1]),0.1, fc = 'w',ec='k')
ax.add_artist(a)
b = plt.Circle((pos2[0],pos2[1]),0.1, fc = 'w',ec='k')
ax.add_artist(b)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
print('Here')
plt.show()
print('Here')
#%% Coulomb


with plt.xkcd(): 
    fig = plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
i = 0
pos1 = [x1, y1s[i]]
pos2 = [x2, y2s[i]]
r = ((((pos1[0]-pos2[0])** 2) + ((pos1[1]-pos2[1])** 2))**0.5)/2
dy = 1/r**2
plt.arrow(pos1[0], pos1[1], 0, -dy, width = 0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') #
plt.arrow(pos2[0], pos2[1], 0, dy, width =0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') # ,length_includes_head=True
plt.text(.2, pos1[1] - dy / 2, 'F',fontsize=20)
plt.text(.2, pos2[1] + dy / 2, 'F',fontsize=20)
plt.xlim((-1.1,1.1))
plt.ylim((-2.5,2.5))
ax = plt.gca()
a = plt.Circle((pos1[0],pos1[1]),0.1, fc = 'w',ec='k')
ax.add_artist(a)
b = plt.Circle((pos2[0],pos2[1]),0.1, fc = 'w',ec='k')
ax.add_artist(b)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
i = -1
    
plt.subplot(1,2,2)
pos1 = [x1, y1s[i]]
pos2 = [x2, y2s[i]]
r = ((((pos1[0]-pos2[0])** 2) + ((pos1[1]-pos2[1])** 2))**0.5)/2
dy = 1/r**2
plt.arrow(pos1[0], pos1[1], 0, -dy, width = 0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') #
plt.arrow(pos2[0], pos2[1], 0, dy, width =0.1*dy ,length_includes_head=True,  color=cols(r_min / r),ec='k') # ,length_includes_head=True
plt.text(.2, pos1[1] - dy / 2, 'F',fontsize=20)
plt.text(.2, pos2[1] + dy / 2, 'F',fontsize=20)
plt.xlim((-1.1,1.1))
plt.ylim((-2.5,2.5))
ax = plt.gca()
a = plt.Circle((pos1[0],pos1[1]),0.1, fc = 'w',ec='k')
ax.add_artist(a)
b = plt.Circle((pos2[0],pos2[1]),0.1, fc = 'w',ec='k')
ax.add_artist(b)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#%% Hydrogren

with plt.xkcd(): 
    fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
# Say, "the default sans-serif font is COMIC SANS"
###
ax = plt.gca()
a = plt.Circle((0,0.11),0.1, fc = 'w',ec='k')
plt.text(0, 0.11, 'P',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(a)
b = plt.Circle((0,-0.11),0.1, fc = 'w',ec='k')
ax.add_artist(b)
plt.text(0, -0.11, 'N',fontsize=20,horizontalalignment='center', verticalalignment='center',)
c = plt.Circle((0,0),0.23, fill = False,ec='k')
ax.add_artist(c)
d = plt.Circle((0,0),0.9, fill = False,ec='k')
plt.text(0, 0.9, 'e',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(d)
e = plt.Circle((0,0.9),0.05, fc = 'w',ec='k')
ax.add_artist(e)
plt.xlim((-1,1))
plt.ylim((-1,1))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#plt.show()   
plt.subplot(1,2,2)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
blue= colorFader('blue','white',0.5)#blue
green=colorFader('green','white',0.5)
red=colorFader('red','white',0.5)

ax = plt.gca()
ax.add_artist(plt.Circle((0,0.11),0.1, fill = False,ec='k'))
x1s = np.linspace(0,-0.045) 
y1s =  np.linspace(0.16,0.07)
x2s = np.linspace(0,0.045) 
y2s =  np.linspace(0.16,0.07)
x3s = np.linspace(-0.045,0.045) 
y3s =  np.linspace(0.07,0.07)    
for i in range(len(x1s)):
    plt.scatter(x1s[i], y1s[i], c=colorFader(blue,green,i/len(x1s)), marker='.', s=500* np.exp(3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x2s[i], y2s[i], c=colorFader(blue,red,i/len(x2s)), marker='.', s=500* np.exp(3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x3s[i], y3s[i], c=colorFader(green,red,i/len(x2s)), marker='.', s=500* np.exp(3*abs((i/len(x1s)) - 1/2)))
        
ax.add_artist(plt.Circle((0,0.16),0.025, fc = blue,ec='k')) #, alpha=0.5
plt.text(0,0.16, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((0.045,0.07),0.025, fc = red,ec='k'))
plt.text(0.045,0.07, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
#plt.plot([0,0.045],[0.16,0.07],c='r',)
ax.add_artist(plt.Circle((-0.045,0.07),0.025, fc = green,ec='k'))
plt.text(-0.045,0.07, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((0,-0.11),0.1,  fill = False,ec='k'))
    
x1s = np.linspace(0,-0.045) 
y1s =  np.linspace(-0.16,-0.07)
x2s = np.linspace(0,0.045) 
y2s =  np.linspace(-0.16,-0.07)
x3s = np.linspace(-0.045,0.045) 
y3s =  np.linspace(-0.07,-0.07)    
for i in range(len(x1s)):
    plt.scatter(x1s[i], y1s[i], c=colorFader(blue,red,i/len(x1s)), marker='.', s=500 * np.exp(3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x2s[i], y2s[i], c=colorFader(blue,green,i/len(x2s)), marker='.', s=500 * np.exp(3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x3s[i], y3s[i], c=colorFader(red,green,i/len(x2s)), marker='.', s=500 * np.exp(3*abs((i/len(x1s)) - 1/2)))  

    
ax.add_artist(plt.Circle((0,-0.16),0.025, fc = blue,ec='k'))
plt.text(0,-0.16, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((0.045,-0.07),0.025, fc = green,ec='k'))
plt.text(0.045,-0.07, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((-0.045,-0.07),0.025, fc = red,ec='k'))
plt.text(-0.045,-0.07, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    
    #ax.add_artist(plt.Circle((0,0),0.23, fill = False,ec='k'))
plt.xlim((-0.25,0.25))
plt.ylim((-0.25,0.25))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#fig.savefig('conf.png', transparent=True)
#%%

with plt.xkcd(): 
    fig = plt.figure(figsize=(12,8))
frames = 40
red=colorFader('red','white',0.5)
antired=colorFader('cyan','white',0.5)
#new_frame = int(frames/2)
new_frame=15
q1 = np.linspace(1,3.8,frames) 
q2 = np.linspace(-1,-3.8,frames)
q3 = np.linspace(-0.5,2.8,frames)
q4 = np.linspace(0.5,-2.8,frames)
qs = 0.25
def string(i):
    ax = plt.gca()
    if(i < new_frame):
        xs = np.linspace(q1[i],q2[i])
        for j in range(len(xs)):
            plt.scatter(0,xs[j], c=colorFader(red,antired,j/len(xs)), marker='.', s=500* np.exp(3*abs((j/len(xs)) - 1/2)))
    else:
        x1s = np.linspace(q1[i],q3[i])
        x2s = np.linspace(q2[i],q4[i])
        for j in range(len(x1s)):
            plt.scatter(0,x1s[j], c=colorFader(red,antired,j/len(x1s)), marker='.', s=500* np.exp(3*abs((j/len(x1s)) - 1/2)))
            plt.scatter(0,x2s[j], c=colorFader(antired,red,j/len(x2s)), marker='.', s=500* np.exp(3*abs((j/len(x2s)) - 1/2)))
        ax.add_artist(plt.Circle((0,q3[i]),qs, fc = antired,ec='k'))
        plt.text(0,q3[i], '$\\bar{q}$',fontsize=20,horizontalalignment='center', verticalalignment='center',)
        ax.add_artist(plt.Circle((0,q4[i]),qs, fc = red,ec='k'))
        plt.text(0,q4[i], '$q$',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    ax.add_artist(plt.Circle((0,q1[i]),qs, fc = red,ec='k'))
    plt.text(0,q1[i], '$q$',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    ax.add_artist(plt.Circle((0,q2[i]),qs, fc = antired,ec='k'))
    plt.text(0,q2[i], '$\\bar{q}$',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    plt.xlim([-2,2])
    plt.ylim([-4,4])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.title('Confined',fontsize=30)
    return plt.plot(np.NaN,np.NaN)
plt.subplot(1,3,1)
string(0)
plt.subplot(1,3,2)
string(14)
plt.subplot(1,3,3)
string(20)
#ns = np.arange(frames)
#ns = np.append(ns, ns[-2:0:-1]) 
#anim = FuncAnimation(fig,string, frames=ns, blit=True, interval = 10)
#anim.save('quark.gif', writer='Pillow')

#%% Deconfined
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,8))

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
blue= colorFader('blue','white',0.5)#blue
green=colorFader('green','white',0.5)
red=colorFader('red','white',0.5)
ax = plt.gca()
    
x1s = np.linspace(-0.21,-0.16,100) #blue red
y1s =  np.linspace(0.2,0.15,100) #blue red
    
x2s = np.linspace(0.1,0.14,100)  #blue green
y2s =  np.linspace(0.2,0.14,100) #blue green
    
x3s = np.linspace(-0.15,-0.1,100)  #red green
y3s =  np.linspace(0.07,0.07,100)    #red green
for i in range(len(x1s)):
    plt.scatter(x1s[i], y1s[i], c=colorFader(blue,red,i/len(x1s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x2s[i], y2s[i], c=colorFader(blue,green,i/len(x2s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x3s[i], y3s[i], c=colorFader(red,green,i/len(x2s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2))) 
        
x1s = np.linspace(-0.16,-0.1,100) #blue red
y1s =  np.linspace(-0.14,-0.2,100) #blue red
    
x2s = np.linspace(0.14,0.09,100)  #blue green
y2s =  np.linspace(-0.14,-0.07,100) #blue green
    
x3s = np.linspace(-0.05,0.05,100)  #red green
y3s =  np.linspace(-0.15,-0.1,100)    #red green
for i in range(len(x1s)):
    plt.scatter(x1s[i], y1s[i], c=colorFader(blue,red,i/len(x1s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x2s[i], y2s[i], c=colorFader(blue,green,i/len(x2s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x3s[i], y3s[i], c=colorFader(red,green,i/len(x2s)), marker='.', s=5000 * np.exp(-3*abs((i/len(x1s)) - 1/2))) 
    
ax.add_artist(plt.Circle((0.1,0.2),0.025, fc = blue,ec='k')) #, alpha=0.5
plt.text(0.1,0.2, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((-0.21,0.07),0.025, fc = red,ec='k'))
plt.text(-0.21,0.07, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((-0.045,0.07),0.025, fc = green,ec='k'))
plt.text(-0.045,0.07, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)


ax.add_artist(plt.Circle((0.14,-0.14),0.025, fc = blue,ec='k'))
plt.text(0.14,-0.14, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((0.09,-0.07),0.025, fc = green,ec='k'))
plt.text(0.09,-0.07, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((-0.1,-0.18),0.025, fc = red,ec='k'))
plt.text(-0.1,-0.18, 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    
ax.add_artist(plt.Circle((-0.05,-0.05),0.025, fc = blue,ec='k'))
plt.text(-0.05,-0.05, 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
    
plt.xlim((-0.25,0.25))
plt.ylim((-0.25,0.25))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#%%
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
ax = plt.gca()
xs = np.linspace(0,100)
    #ys = -np.log1p(np.abs(50 - xs))*np.sign(50-xs)
ys = 1/ (np.abs(50-xs)+1)
xs = np.linspace(0,100)
ys = -np.log1p(np.abs(50 - xs))*np.sign(50-xs)
ax.plot(xs,ys,'b-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('T',fontsize = 20)
ax.set_ylabel('E',rotation = 0, labelpad=10,fontsize = 20) 
ax.set_title('Cross over',fontsize = 20)
    
    
    
plt.subplot(1,2,2)
ax = plt.gca()
Tmax = 100; Tc = Tmax / 2
xs = np.linspace(0,Tmax,500)
ys = -np.log1p(np.abs(Tc - xs))*np.sign(Tc-xs) - 5*np.sign(Tc-xs)
if(np.sum(ys == Tc) == 1): print('here')
ax.plot(xs[xs < 50],ys[xs<50],'b-')
ax.plot([max(xs[xs < 50]),min( xs[xs > 50])], [max(ys[xs < 50]),min(ys[xs>50])],'b--')
ax.plot(xs[xs > 50],ys[xs>50],'b-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('T',fontsize = 20)
ax.set_ylabel('E',rotation = 0, labelpad=10,fontsize = 20) 
ax.set_title('First order',fontsize = 20)
plt.show()

#%% Deconf conf potential
def fun(x):
    s=  - (((x>=0)*0.99*x)**4 - 11.3*(x**2)) + ((x>=0)*0.1*(x-1)**6  )
    size = x[-1] - x[0]
    print(s.max())
    s = (s/(s.max() - s.min()))*size
    return s
x =np.arange(-2, 4.8, 0.05)
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
blue= colorFader('blue','white',0.5)#blue
green=colorFader('green','white',0.5)
red=colorFader('red','white',0.5)
antired=colorFader('cyan','white',0.5)
y = fun(x)
plt.plot(x,y.flatten())
ax = plt.gca()
qs = 0.25
q1 = [4.3,2]
q2 = [3.5,4]
x1s = np.linspace(4.2 ,3.7)#red-> blue
y1s =  np.linspace(3.4,2.6)  
ax.add_artist(plt.Circle((q2[0],q2[1]),qs, fc = red,ec='k'))
plt.text(q2[0],q2[1], 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((q1[0],q1[1]),qs, fc = blue,ec='k'))
plt.text(q1[0],q1[1], 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
q4= [0,2.5 - 1.2*(3**0.5)/2] # red
q5 = [0.6,2.5] #blue
q6 = [-0.6,2.5] # green
x4s = np.linspace(q4[0],q5[0])#red-> blue
y4s =  np.linspace(q4[1],q5[1])  
x5s = np.linspace(q5[0],q6[0]) #blue->green
y5s =  np.linspace(q5[1],q6[1])  
x6s = np.linspace(q6[0],q4[0]) #green->red
y6s =  np.linspace(q6[1],q4[1])  
for i in range(len(x1s)):
    plt.scatter(x1s[i], y1s[i], c=colorFader(green,red,i/len(x4s)), marker='.', s=3000 * np.exp(-3*abs((i/len(x1s)) - 1/2)))
    plt.scatter(x4s[i], y4s[i], c=colorFader(red,blue,i/len(x4s)), marker='.', s=500* np.exp(3*abs((i/len(x4s)) - 1/2)))
    plt.scatter(x5s[i], y5s[i], c=colorFader(blue,green,i/len(x4s)), marker='.', s=500* np.exp(3*abs((i/len(x4s)) - 1/2)))
    plt.scatter(x6s[i], y6s[i], c=colorFader(green,red,i/len(x4s)), marker='.', s=500* np.exp(3*abs((i/len(x4s)) - 1/2)))
ax.add_artist(plt.Circle((q4[0],q4[1]),qs, fc = red,ec='k')) #, alpha=0.5
plt.text(q4[0],q4[1], 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax.add_artist(plt.Circle((q5[0],q5[1]),qs, fc = blue,ec='k')) #, alpha=0.5
plt.text(q5[0],q5[1], 'u',fontsize=20,horizontalalignment='center', verticalalignment='center',)    
ax.add_artist(plt.Circle((q6[0],q6[1]),qs, fc = green,ec='k')) #, alpha=0.5
plt.text(q6[0],q6[1], 'd',fontsize=20,horizontalalignment='center', verticalalignment='center',)  
    
    #ax.add_artist(plt.Circle((q4[0],q4[1]),qs, fc = blue,ec='k')) #, alpha=0.5
plt.text(3.8,5.4, 'Deconfined',fontsize=20,horizontalalignment='center', verticalalignment='center',)    
plt.text(0,5.4, 'Confined',fontsize=20,horizontalalignment='center', verticalalignment='center',)    
    


plt.ylabel('$V(\phi)$',fontsize=20)
plt.xlabel('$\phi$',fontsize=20)
plt.xlim([x.min(),x.max()])
plt.ylim([0-0.1,y.max()])
plt.yticks([])
plt.xticks([])
plt.show()
#%% This is a particle

def Gaussian(x,y, mnx,mny, amp, std):
    r = ((x - mnx)**2 + (y-mny)**2)**0.5
    G1 = np.exp(-((r)**2.)/(2 * (std**2.)))  * amp
    return G1

with plt.xkcd(): 
    fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1, 3, 1)
ax.add_artist(plt.Circle((0,0),0.05, fc = 'k',ec='k'))
plt.xlim((-1,1))
plt.ylim((-1,1))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#plt.text(0.1,0.0, 'q',fontsize=20,horizontalalignment='left', verticalalignment='center',)
plt.text(0,0.3, 'This is a particle',fontsize=20,horizontalalignment='center', verticalalignment='center',)
ax = fig.add_subplot(1,3,2,projection='3d')

mnx =0; mny=0;
std = 0.1
amp = 1
grid = np.linspace(-1,1)
xx, yy = np.meshgrid(grid, grid)
zz = Gaussian(xx,yy, mnx,mny, amp, std)
ax.plot_surface(xx, yy, zz,cmap=mpl.cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.xlim((-1,1))
plt.ylim((-1,1))
ax.set_zlim(0, 0.8)
ax.text(0, -1.3, 0, 'This is a field',  'x',horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.set_axis_off()
ax = fig.add_subplot(1,3,3,projection='3d')
mnx =0; mny=0;
std = 0.1
amp = 1
lgrid = np.linspace(-1,1,11)
lxx, lyy = np.meshgrid(lgrid, lgrid)
lzz = Gaussian(lxx,lyy, mnx,mny, amp, std)
ax.plot_surface(xx, yy, zz,cmap=mpl.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha =0.1,zorder=1)
for X,Y,Z in zip(lxx,lyy,lzz):
    for x, y, z in zip(X,Y,Z):
        ax.scatter(x, y, z,marker='o',c=mpl.cm.coolwarm(z),zorder=2) # 
plt.xlim((-1,1))
plt.ylim((-1,1))
ax.set_zlim(0, 0.8)
ax.text(0, -1.5, 0, 'This is a field\n on a lattice',  'x',horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.set_axis_off()
plt.show()

#%% This is a lattice
xs = np.linspace(0,-1,4)

#xx, yy = np.meshgrid(x, x)
with plt.xkcd(): 
    fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,3,1,projection='3d')
for j, y in enumerate(xs):        
    for k, z in enumerate(xs):
        for i, x in enumerate(xs):
            ax.plot([x,x],[xs[j],xs[min(len(xs)-1, j+1)]],[z,z],c='darkorange')
            ax.plot([xs[i],xs[min(len(xs)-1, i+1)]],[y,y],[z,z],c='darkorange')
            ax.plot([x,x],[y,y],[xs[k],xs[min(len(xs)-1, k+1)]],c='darkorange' )
            ax.scatter(x,  y,z, c = 'b',marker='o')
    ax.text(-0.5, .15, 0, 'This is a lattice',  'x',horizontalalignment='center', verticalalignment='center',fontsize=20) # 

ax.set_axis_off()

ax = fig.add_subplot(1,3,2)
for j, y in enumerate(xs): 
    plt.plot(xs,y*np.ones_like(xs),'darkorange', zorder=1)
    plt.plot(y*np.ones_like(xs),xs,'darkorange', zorder=1)
    for i, x in enumerate(xs):
        ax.scatter(x,  y, c = 'b',marker='o',s=100,zorder=3)
ax.text(-0.5, .1, 'Gauge field + fermions',  horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.text(-0.63, -0.25, 'quark',  horizontalalignment='left', verticalalignment='center',fontsize=20,color='b')
ax.text(-0.5, -0.75, 'gluon',  horizontalalignment='center', verticalalignment='center',fontsize=20,color='darkorange')
a = xs[1] - xs[0]
plt.arrow(xs[1], xs[2], 2*a/3,0, width = 0.001,head_width = 0.05, length_includes_head=True, color='darkorange')
ax.set_axis_off()

ax = fig.add_subplot(1,3,3)
for j, y in enumerate(xs): 
    plt.plot(xs,y*np.ones_like(xs),'darkorange', zorder=1)
    plt.plot(y*np.ones_like(xs),xs,'darkorange', zorder=1)
ax.text(-0.5, .1, 'Pure gauge',  horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.text(-0.5, -0.75, 'gluon',  horizontalalignment='center', verticalalignment='center',fontsize=20,color='darkorange')
a = xs[1] - xs[0]
plt.arrow(xs[1], xs[2], 2*a/3,0, width = 0.001,head_width = 0.05, length_includes_head=True, color='darkorange')
ax.set_axis_off()
ax.set_xlim((-1.05, 0.05))
ax.set_ylim((-1.05, 0.05))

#%% Wilson loop
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,4))
xs = np.linspace(0,-1,8)
ax = fig.add_subplot(1,2,1)
for j, y in enumerate(xs): 
    plt.plot(xs,y*np.ones_like(xs),'darkorange', zorder=1)
    plt.plot(y*np.ones_like(xs),xs,'darkorange', zorder=1)
a = xs[1] - xs[0]
path = [[1,0],[0,-1],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0],[0,1],[-1,0],[-1,0],[-1,0],[0,-1],[0,-1],[-1,0],[0,-1]]
i = 1; j = 2;
for p in path: 
    plt.arrow(xs[i], xs[j], p[0]*a,p[1]*a, width = 0.001,head_width = 0.05, length_includes_head=True, color='k',zorder = 2)
    i+=p[0];j+=p[1]
ax.set_axis_off()
ax.set_xlim((-1.05, 0.05))
ax.set_ylim((-1.05, 0.05))
ax.set_axis_off()
xs = np.linspace(0,-1,4)
ax = fig.add_subplot(1,2,2,projection='3d')
path = [[0,0,3,1,0,0],[1,0,3,0,0,-1],[1,0,2,0,0,-1],[1,0,1,0,0,-1],[1,0,0,0,1,0],
            [1,1,0,0,1,0],[1,2,0,1,0,0],[2,2,0,0,0,1],[2,2,1,0,1,0],[2,3,1,-1,0,0],
            [1,3,1,0,-1,0],[1,2,1,0,0,1],[1,2,2,0,1,0],[1,3,2,-1,0,0],[0,3,2,0,-1,0],
            [0,2,2,0,0,1],[0,2,3,0,-1,0],[0,1,3,0,-1,0]]


for j, y in enumerate(xs):        
    for k, z in enumerate(xs):
        for i, x in enumerate(xs):
                
            ax.plot([x,x],[xs[j],xs[min(len(xs)-1, j+1)]],[z,z],c='darkorange',zorder =3 )
            ax.plot([xs[i],xs[min(len(xs)-1, i+1)]],[y,y],[z,z],c='darkorange',zorder =3)
            ax.plot([x,x],[y,y],[xs[k],xs[min(len(xs)-1, k+1)]],c='darkorange',zorder =3)
            for p in path:    
                if(i==p[0]+p[3] and p[3] >= 0)and(j==p[1])and(k==p[2]):
                    u = -(xs[p[0]]-xs[p[0]+p[3]])
                    v = -(xs[p[1]]-xs[p[1]+p[4]])
                    w = -(xs[p[2]]-xs[p[2]+p[5]])
                    ax.quiver(x-u,y,z,u,v,w, ec='k')#,zorder=0 
                if(i==p[0] and p[3] < 0)and(j==p[1])and(k==p[2]):
                    u = -(xs[p[0]]-xs[p[0]+p[3]])
                    v = -(xs[p[1]]-xs[p[1]+p[4]])
                    w = -(xs[p[2]]-xs[p[2]+p[5]])
                    ax.quiver(x,y,z,u,v,w, ec='k') 
ax.set_axis_off()
#%%

def data_for_cylinder_along_z(center_x,center_y,radius,height_z, zpoints,tpoints):
    z = np.linspace(0, height_z, zpoints)
    theta = np.linspace(0, 2*np.pi, tpoints)
    theta_grid, x_grid=np.meshgrid(theta, z)
    z_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# for x,y,z in zip(Xc,Yc,Zc):
#     ax.scatter(Xc, Yc, Zc, c='k')
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1,50,50)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1)
Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1,8,8)
for xs,ys,zs in zip(Xc.T,Yc.T,Zc.T):
    ax.plot(xs[ys > 0.22],ys[ys > 0.22],zs[ys > 0.22], c='darkorange')
for xs,ys,zs in zip(Xc,Yc,Zc):
    ax.plot(xs,ys,zs, c='darkorange')
n = 3
xs = Xc[n]; ys = Yc[n]; zs = Zc[n]; 
for i in range(len(xs) - 1):
    x = xs[i]; y = ys[i]; z = zs[i]
    u = xs[i+1] - xs[i]; v = ys[i+1] - ys[i]; w = zs[i+1] - zs[i]
    ax.quiver(x,y,z,u,v,w, ec='k')
i=0
u = xs[i+1] - xs[i]; v = ys[i+1] - ys[i]; w = zs[i+1] - zs[i]
x = xs[i] ; y = ys[i]+ v/2; z = zs[i] #+ 0.01
ax.text(x,y,z, '$l_p$',  (u,v,w),horizontalalignment='center', verticalalignment='center',fontsize=20)
for xs,ys,zs in zip(Xc.T,Yc.T,Zc.T):
    ax.plot(xs[ys <= 0.22],ys[ys <= 0.22],zs[ys <= 0.22], c='darkorange',zorder =5)
Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1,1,25)
ax.plot_surface(Xc, Yc, Zc, alpha=0.3)
for xs,ys,zs in zip(Xc,Yc,Zc):
    ax.plot(xs+0.12,ys,zs, c='b')
    i= 20
    x = xs[i]; y = ys[i]; z = zs[i]
    u = xs[i+2] - xs[i]; v = ys[i+2] - ys[i]; w = zs[i+2] - zs[i]
    wid =3
    ax.quiver(x+0.12,y,z,u,v,w, ec='b',linewidths=wid)
ax.quiver(0,0.17,0.15,0.12,0,0, ec='b',linewidths=wid)
ax.text(0.06,0.15,0.15, 'Space',  'x',horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.text(0.12, 0.18, 0.22, 'Time',  (u, v, w),horizontalalignment='center', verticalalignment='center',fontsize=20)
ax.set_axis_off()
plt.show()
xs = np.linspace(0,-1,8)
with plt.xkcd(): 
    fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
for j, y in enumerate(xs): 
    plt.plot(xs,y*np.ones_like(xs),'darkorange', zorder=1)
    plt.plot(y*np.ones_like(xs),xs,'darkorange', zorder=1)
a = xs[1] - xs[0]
path = [[1,0],[0,1],[-1,0],[0,-1]]
i = 1; j = 2;
for p in path: 
    plt.arrow(xs[i], xs[j], p[0]*a,p[1]*a, width = 0.001,head_width = 0.05, length_includes_head=True, color='k',zorder = 2)
    i+=p[0];j+=p[1]
ax.text(-0.21, -.35, '$u_p$',  horizontalalignment='center', verticalalignment='center',fontsize=25)
ax.set_axis_off()
ax.set_xlim((-1.05, 0.05))
ax.set_ylim((-1.05, 0.05))

#%%
def plot_lattice(N):
    xs = np.linspace(0,-1,N)
    with plt.xkcd(): 
        fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(projection='3d')
    for j, y in enumerate(xs):        
        for k, z in enumerate(xs):
            for i, x in enumerate(xs):
                ax.plot([x,x],[xs[j],xs[min(len(xs)-1, j+1)]],[z,z],c='darkorange' )
                ax.plot([xs[i],xs[min(len(xs)-1, i+1)]],[y,y],[z,z],c='darkorange')
                ax.plot([x,x],[y,y],[xs[k],xs[min(len(xs)-1, k+1)]],c='darkorange')
    ax.set_axis_off()
    fig.savefig(f'Lat{N}.png')
plot_lattice(4)
plot_lattice(6)
plot_lattice(8)