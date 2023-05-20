#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:30:48 2023

@author: rajsah
"""
import numpy as np
import matplotlib.pyplot as plt
#import sys
from scipy.interpolate import make_interp_spline

#plt.rc('font',family='Times New Roman')
plt.rc('font',family='Callibri')

def interpolation(x,y,pts,a): #note a=3 is good
   X_Y_Spline = make_interp_spline(x, y,k=a)
   X_ = np.linspace(x.min(), x.max(), pts)
   Y_ = X_Y_Spline(X_)
   return X_,Y_

def noise(x,y,thresh): # this function removes any value in the data > a
    PDx1=[];PDy1=[]
    for i in range(len(x)):
        if abs(y[i])<=thresh:
            PDx1.append(x[i])
            PDy1.append(y[i])
    x2=np.array(PDx1)
    y2=np.array(PDy1)
    return x2,y2

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def fig2b():
    shift_2b= -0.04549
    path2b='/Users/rajsah/OneDrive - Temple University/work-DFT/catalysis/'
    path2b_dir='NiO2/dos-plot/supercell/GGA+U-1k'
    data1=np.loadtxt(""+str(path2b)+""+str(path2b_dir)+"/TDOS.dat",skiprows=1)
    data2=np.loadtxt(""+str(path2b)+""+str(path2b_dir)+"/PDOS_A11_UP.dat",skiprows=1)
    data3=np.loadtxt(""+str(path2b)+""+str(path2b_dir)+"/PDOS_A11_DW.dat",skiprows=1)
    ''' extracting PDOS '''
    pdu=data2[:,5]+data2[:,6]+data2[:,7]+data2[:,8]+data2[:,9]
    pdd=data3[:,5]+data3[:,6]+data3[:,7]+data3[:,8]+data3[:,9]
    #noise removal on TDOS
    x_u_tdn,y_u_tdn=noise(data1[:,0],data1[:,1],150)
    x_d_tdn,y_d_tdn=noise(data1[:,0],data1[:,2],150)
    # interpolation of TDOS
    x_tu,y_tu = interpolation(x_u_tdn,y_u_tdn,5000,3)
    x_td,y_td = interpolation(x_d_tdn,y_d_tdn,5000,3)
    
    #noise removal on PDOS
    x_pu_n,y_pu_n = noise(data2[:,0],pdu,6)
    x_pd_n,y_pd_n = noise(data3[:,0],pdd,6)
    # interpolation on PDOS
    x_pu,y_pu = interpolation(x_pu_n,y_pu_n,5000,3) 
    x_pd,y_pd = interpolation(x_pd_n,y_pd_n,5000,3)
   
    #lets merge the data here
    xpu_=np.array(list(x_pu)+list(x_pd))
    ypu_=np.array(list(y_pu)+list(y_pd))
    x_td_=np.array(list(x_tu)+list(x_td))
    y_td_=np.array(list(y_tu)+list(y_td))

    #plt.plot(xpu_-shift_2b,ypu_,color='b',linewidth=3,label='$Mn_d$')
    plt.plot(xpu_-shift_2b,smooth(ypu_,12),color='blue',linewidth=3,label='$Mn_d$')
    #plt.fill_between(x_td_-shift_2b,y_td_/2,alpha=0.4,color='grey',label='Total')
    plt.fill_between(x_td_-shift_2b,smooth(y_td_,12)/32,alpha=0.5,color='red',label='Total')
    """ The weirdness in Total density was cured by removing extra data in Conduction band """
    plt.xlim(-6.5,3.5)
    plt.ylim(-4.5,4.5)
    plt.xticks(np.arange(-6, 4, step=1),fontsize=12,fontweight='bold')
    plt.yticks(np.arange(-4, 4.5, step=2),fontsize=12,fontweight='bold')
    plt.title("NiO$_2$ : Supercell DOS",fontsize=12,fontweight='bold')
    plt.xlabel('Energy difference $\epsilon$-$\epsilon$$_{CBM}$ (eV)',fontsize=12,fontweight='bold')
    #plt.xlabel('Energy(no shift) (eV)',fontsize=12,fontweight='bold')

    plt.ylabel('DOS (arb. units)',fontsize=12,fontweight='bold')
    plt.legend(loc='lower left',fontsize=12,frameon=False)
    plt.axvline(0, color='black',linestyle='--')
    plt.axhline(0, color='black',linestyle='solid')
    axes=plt.gca()
    axes.set_aspect(0.6)
    plt.text(1.5,3.8,r'SPIN-UP',fontsize=12,fontweight='bold')
    plt.text(1,-4,r'SPIN-DOWN',fontsize=12,fontweight='bold')
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in')
    #plt.savefig(""+str(path2b)+""+str(path2b_dir)+"/fig2B-Mn11.pdf")
    plt.show()


def fig2c():
    shift_2c= -0.04549
    ''' setting path and importing datas'''
    path2c='/Users/rajsah/OneDrive - Temple University/work-DFT/catalysis/'
    path2c_dir='NiO2/dos-plot/supercell/GGA+U-1k'
    # lets work on TDOS
    data_tot=np.loadtxt(""+str(path2c)+""+str(path2c_dir)+"/TDOS.dat",skiprows=1)
    Tx_u,Ty_u=noise(data_tot[:,0],data_tot[:,1],150) 
    Tx_d,Ty_d=noise(data_tot[:,0],data_tot[:,2],150)
    Tx_ui,Ty_ui=interpolation(Tx_u,Ty_u,1500,3)
    Tx_di,Ty_di=interpolation(Tx_d,Ty_d,1500,3)
    Tx = np.array(list(Tx_ui)+list(Tx_di))
    Ty = np.array(list(Ty_ui)+list(Ty_di))   
    #plt.fill_between(Tx-shift_2c,Ty/32,alpha=0.5,color='grey',label='Total')
    plt.xlim(-6.5,3.5)
    #plt.ylim(-6,6)
    plt.ylim(-5.2,5.2)
    plt.fill_between(Tx-shift_2c,smooth(Ty,3)/32,alpha=0.5,color='red',label='Total')
    
    # lets work on PDS of Mn atom with polaron
    data1_up=np.loadtxt(""+str(path2c)+""+str(path2c_dir)+"/PDOS_A27_UP.dat",skiprows=1)
    data1_dn=np.loadtxt(""+str(path2c)+""+str(path2c_dir)+"/PDOS_A27_DW.dat",skiprows=1)   
    pdu_1=data1_up[:,5]+data1_up[:,6]+data1_up[:,7]+data1_up[:,8]+data1_up[:,9]   
    pdd_1=data1_dn[:,5]+data1_dn[:,6]+data1_dn[:,7]+data1_dn[:,8]+data1_dn[:,9]
    pu_xn,pu_yn=noise(data1_up[:,0],pdu_1,6) #noise removal
    pu_xi,pu_yi=interpolation(pu_xn,pu_yn,550,3) # interpolation 
    pd_xn,pd_yn=noise(data1_up[:,0],pdd_1,6) #noise removal
    pd_xi,pd_yi=interpolation(pd_xn,pd_yn,550,3) # interpolation  
   
   # lets merge the data 
    Mn3_dx=np.array(list(pu_xi)+list(pd_xi)) #merging x
    Mn3_dy=np.array(list(pu_yi)+list(pd_yi)) #merging y
    
    #plt.plot(Mn3_dx-shift_2c,Mn3_dy,color='b',linewidth=2.5,label='Mn$_d$')
    plt.plot(Mn3_dx-shift_2c,smooth(Mn3_dy,2),color='b',linewidth=2.5,label='Mn$_d$')
  
    
    plt.xlabel('Energy difference $\epsilon$-$\epsilon$$_{CBM}$ (eV)',fontsize=12,fontweight='bold')
    plt.ylabel('DOS (arb. units)',fontsize=12,fontweight='bold')
    plt.xticks(np.arange(-6, 4, step=1),fontsize=12,fontweight='bold')
    plt.yticks(np.arange(-5, 6, step=5),fontsize=12,fontweight='bold')
    plt.xlim(-6.5,3.5)
    #plt.ylim(-6,6)
    plt.ylim(-5.2,5.2)
    plt.axvline(0, color='black',linestyle='--')
    plt.axhline(0, color='black',linestyle='solid')
    #plt.legend(label=labels,loc='lower left',fontsize=12,frameon=False)
    plt.legend(loc='lower left',fontsize=12,frameon=False)
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in')
    axes=plt.gca()
    axes.set_aspect(0.5)
    plt.text(1,4.5,r'SPIN-UP',fontsize=12,fontweight='bold')
    plt.text(1,-4.5,r'SPIN-DOWN',fontsize=12,fontweight='bold')
    #plt.savefig(""+str(path1)+""+str(path1a)+"/fig3a-may7-new.pdf")
    plt.savefig(""+str(path2c)+""+str(path2c_dir)+"/fig2c.pdf")
    plt.show()

def fig3():
    shift_3=0.08940
    ''' setting path and importing datas'''
    path1='/Users/rajsah/OneDrive - Temple University/work-DFT/catalysis/MnO2/vasp/dos-plot/'
    path1a='supercell/with-k/1k/May5'
    
    ''' work on PDOS of polaron i.e. Mn(+3) '''
    data1_up=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A11_UP.dat",skiprows=1)
    data1_dn=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A11_DW.dat",skiprows=1)   
    pdu_1=data1_up[:,5]+data1_up[:,6]+data1_up[:,7]+data1_up[:,8]+data1_up[:,9]   
    pdd_1=data1_dn[:,5]+data1_dn[:,6]+data1_dn[:,7]+data1_dn[:,8]+data1_dn[:,9]
    p_xn,p_yn=noise(data1_up[:,0],pdu_1,15) #noise removal
    p_xi,p_yi=interpolation(p_xn,p_yn,500,3) # interpolation 
    #plt.plot(p_xi-shift_3,p_yi,color='r',linewidth=2.5,label='$Mn_d$${III}$')
    plt.plot(p_xi-shift_3,smooth(p_yi,2),color='r',linewidth=2.5,label='$Mn_d$${III}$')


    '''work on DPOS of a Mn(+4) in layer with polar  '''
    data2_up=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A1_UP.dat",skiprows=1)
    data2_dn=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A1_DW.dat",skiprows=1)   
    pdu_2=data2_up[:,5]+data2_up[:,6]+data2_up[:,7]+data2_up[:,8]+data2_up[:,9]   
    pdd_2=data2_dn[:,5]+data2_dn[:,6]+data2_dn[:,7]+data2_dn[:,8]+data2_dn[:,9]
    pd1_xn,pd1_yn=noise(data2_dn[:,0],pdd_2,10)
    pd1_xi,pd1_yi=interpolation(pd1_xn,pd1_yn,700,3)

 
    '''work on DPOS of Mn(+4) in layer without polar '''
    data3_up=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A17_UP.dat",skiprows=1)
    data3_dn=np.loadtxt(""+str(path1)+""+str(path1a)+"/PDOS_A17_DW.dat",skiprows=1)
    pdu_3=data3_up[:,5]+data3_up[:,6]+data3_up[:,7]+data3_up[:,8]+data3_up[:,9]
    pdd_3=data3_dn[:,5]+data3_dn[:,6]+data3_dn[:,7]+data3_dn[:,8]+data3_dn[:,9]
    pd2_xn,pd2_yn=noise(data3_up[:,0],pdu_3,10)
    pd2_xi,pd2_yi=interpolation(pd2_xn,pd2_yn,700,3)


    '''  lets merge data here to plot Mn_d form Mn(+4) '''
    x_d=np.array(list(pd1_xi)+list(pd2_xi))
    y_d=np.array(list(-pd1_yi)+list(-pd2_yi))

    #plt.plot(x_d-shift_3,y_d,color='b',linewidth=2.5,label='$Mn_d$')
    plt.plot(x_d-shift_3,smooth(y_d,2),color='b',linewidth=2.5,label='$Mn_d$')

    #plt.title(r"Super cell with 1K",fontweight='bold')
    plt.xlabel('Energy difference $\epsilon$-$\epsilon$$_{CBM}$ (eV)',fontsize=12,fontweight='bold')
    plt.ylabel('DOS (arb. units)',fontsize=12,fontweight='bold')
    plt.xticks(np.arange(-6, 4, step=1),fontsize=12,fontweight='bold')
    plt.yticks(np.arange(-5, 6, step=5),fontsize=12,fontweight='bold')
    plt.xlim(-6.5,3.5)
    #plt.ylim(-6,6)
    plt.ylim(-5.2,5.2)
    plt.axvline(0, color='black',linestyle='--')
    plt.axhline(0, color='black',linestyle='solid')
    #plt.legend(label=labels,loc='lower left',fontsize=12,frameon=False)
    plt.legend(loc='lower left',fontsize=12,frameon=False)
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in')
    axes=plt.gca()
    axes.set_aspect(0.5)
    #plt.savefig(""+str(path1)+""+str(path1a)+"/fig2c-may9.pdf")
    plt.show()

# lets call the functions for the figures
#fig2b() 
fig2c()
#fig3()


    
    

