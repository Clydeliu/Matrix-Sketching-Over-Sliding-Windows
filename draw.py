# -*- coding: gb2312 -*-
# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np
# Sampling with replacement
nswmaxsketchsize=np.loadtxt('NSWmaxsketchsize.txt') # Max sketch size (rows)
nswmaxerr=np.loadtxt('NSWmaxerr.txt') # Maximum err
nswaverr=np.loadtxt('NSWaverr.txt') # Average err
nswtime=np.loadtxt('NSWtime.txt') #Running Time
nswtime=nswtime/10 #update cost per item(ms) for SWR. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row
# Sampling without replacement
nsnmaxsketchsize=np.loadtxt('NSNmaxsketchsize.txt') # Max sketch size (rows)
nsnmaxerr=np.loadtxt('NSNmaxerr.txt') # Maximum err
nsnaverr=np.loadtxt('NSNaverr.txt') # Average err
nsntime=np.loadtxt('NSNtime.txt') #Running Time
nsntime=nsntime/10 #update cost per item(ms) for SWOR. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row
# Sampling without replacement-using all rows
tmaxsketchsize=np.loadtxt('TESTmaxsketchsize.txt') # Max sketch size (rows)
tmaxerr=np.loadtxt('TESTmaxerr.txt') # Maximum err
taverr=np.loadtxt('TESTaverr.txt') # Average err
ttime=np.loadtxt('TESTtime.txt') #Running Time
ttime=ttime/10 #update cost per item(ms) for SWOR-ALL. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row
#LM-FD
efdmaxsketchsize=np.loadtxt('EHFDmaxsketchsize.txt') # Max sketch size (rows)
efdmaxerr=np.loadtxt('EHFDmaxerr.txt') # Maximum err
efdaverr=np.loadtxt('EHFDaverr.txt') # Average err
efdtime=np.loadtxt('EHFDtime.txt') #Running Time
efdtime=efdtime/10 #update cost per item(ms) for LM-FD. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row
#DI-FD
dfdmaxsketchsize=np.loadtxt('dFDmaxsketchsize.txt') # Max sketch size (rows)
dfdmaxerr=np.loadtxt('dFDmaxerr.txt') # Maximum err
dfdaverr=np.loadtxt('dFDaverr.txt') # Average err
dfdtime=np.loadtxt('dFDtime.txt') #Running Time
dfdtime=dfdtime/10 #update cost per item(ms) for DI-FD. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row
#Best-Offline
bestmaxsketchsize=np.loadtxt('bestsketchsize.txt') # Max sketch size (rows)
bestmaxerr=np.loadtxt('bestmaxerr.txt') # Maximum err
bestaverr=np.loadtxt("bestaverr.txt") # Average err
besttime=np.loadtxt("besttime.txt") #Running Time
besttime=besttime/10 #update cost per item(ms) for BEST-OFFLINE. 1 s/10000 row = 1000 ms/ 10000 row = 1/10 ms/row

fig1=plt.figure('fig1',figsize=(8,7))
plt.plot(nswmaxsketchsize,nswmaxerr,'bo-',nsnmaxsketchsize,nsnmaxerr,'c^-',tmaxsketchsize,tmaxerr,'k>-',bestmaxsketchsize, bestmaxerr,'rv-',efdmaxsketchsize,efdmaxerr,'mp-',dfdmaxsketchsize,dfdmaxerr,'gx-',markersize=12,linewidth=2)
plt.axis('auto')
plt.tick_params(labelsize=20)
plt.xlabel('Max sketch size (rows)',fontsize=25)
plt.ylabel('Maximum err',fontsize=25)
plt.legend(('Baseline: SWR','Baseline: SWOR','Baseline: SWOR-ALL','BEST(offline)','LM-FD','DI-FD'),loc=1,ncol=2)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
plt.axis([0,300,0,0.73])
fig1.savefig('all-maxerr.eps', dpi=75)

fig2=plt.figure('fig2',figsize=(8,7))
plt.plot(nswmaxsketchsize,nswaverr,'bo-',nsnmaxsketchsize,nsnaverr,'c^-',tmaxsketchsize,taverr,'k>-',bestmaxsketchsize, bestaverr,'rv-',efdmaxsketchsize,efdaverr,'mp-',dfdmaxsketchsize,dfdaverr,'gx-',markersize=12,linewidth=2)
plt.axis('auto')
plt.tick_params(labelsize=20)
plt.xlabel('Max sketch size (rows)',fontsize=25)
plt.ylabel('Average err',fontsize=25)
plt.legend(('Baseline: SWR','Baseline: SWOR','Baseline: SWOR-ALL','BEST(offline)','LM-FD','DI-FD'),loc=1,ncol=2)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
plt.axis([0,300,0,0.51])
fig2.savefig('all-averr.eps', dpi=75)

fig3=plt.figure('fig3',figsize=(8,7))
plt.plot(nswmaxsketchsize,nswtime,'bo-',nsnmaxsketchsize,nsntime,'c^-',tmaxsketchsize,ttime,'k>-',efdmaxsketchsize,efdtime,'mp-',dfdmaxsketchsize,dfdtime,'gx-',markersize=12,linewidth=2)
plt.axis('auto')
plt.tick_params(labelsize=20)
plt.xlabel('Max sketch size (rows)',fontsize=25)
plt.ylabel('Update cost per item (ms)',fontsize=25)
plt.axis([0,300,0,41])
plt.legend(('Baseline: SWR','Baseline: SWOR','Baseline: SWOR-ALL','LM-FD','DI-FD'),loc=1,ncol=2)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
fig3.savefig('all-time.eps', dpi=75)