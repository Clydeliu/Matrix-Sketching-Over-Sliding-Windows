# -*- coding: gb2312 -*-
# -*- coding: gbk -*-
from __future__ import division
import numpy as np
import scipy.io as sio
from scipy import random, linalg, dot, zeros, diag, sqrt, floor, ceil, remainder, log2
import time

data=sio.loadmat('rail2586') # read in data
A=data['Problem'][0,0] 
A=A[7] # need to find where the matrix is stored in data
A=A.T # num_row is much bigger than num_column
A=A.toarray() # convert compressed matrix into 2-dimentional numpy array, denote A the matrix data
#matrix size
n=A.shape[0] # num_row
m=A.shape[1] # num_column
#window size
N=10000 # define the window size for sequence-based window
jg=2500 # define the frequence to measure error between window and sketch, 2500 means measure once 2500 rows passed 
mi=1 # minum row norm in the data
repeattime=20 # define how many times to run for algorithms containing randomness

# fuctions used in sampling
# 3 functions are used to approximate current window's norm with LM methods
def normmerge(block_old,block_new):
    # merge two blocks, block_new is more recent block
    # return merged block
    block_merged=[0,0,0] # initiate a new block
    block_merged[0]=block_old[0] # start time eaquals block_old's
    block_merged[1]=block_new[1] # end time eqauals block_new's
    block_merged[2]=block_old[2]+block_new[2] # norm is the sum
    return block_merged

def normremove(k,N,level):
    # remove rows in level that expires in the window
    global L # current num_level
    global block # current stored blocks where each block contains information of start time, end time, norm of block
    if level!=0: 
        for item in block['levelX'.replace('X',str(level))]: # for every row in level
            if k-item[0]>N: # if row's start time is out of current window
                block['levelX'.replace('X',str(level))].remove(item) # remove it
            else:
                break # if this row's start time is not out of current window, sure later rows are not neither
        if len(block['levelX'.replace('X',str(level))])==0 and level!=1: # if rows in this level is all removed
            L=L-1 # current num_level should minus 1
            normremove(k,N,level-1) # continur to remove rows in level-1 if any expires.

def calnorm():
    global block,underblock, L # current stored blocks, block under_construction , num_level
    window_norm=0 # calculate the norm of current window
    for i in range(L,0,-1):
        if len(block['levelX'.replace('X',str(i))])!=0:
            for item in block['levelX'.replace('X',str(i))]:
                window_norm=window_norm+item[2]
    window_norm=window_norm+underblock[2]
    return window_norm

#sampling
sl1=[3,5,7,10,13,15,20,22,25] # l run for this algorithm
nswmaxerr=[] # stores maximum err for each l
nswaverr=[] # stores average err for each l
nswmaxsketchsize=[] # stores maximum sketch size for each l
nswavsketchsize=[] # stores average sketch size for each l
nswtime=[] # stores running time for each l

#Norm Sampling with replacement
for l in sl1:
    maxerr=[] # stores maximum err for each running for l
    averr=[] # stores average err for each running for l
    maxsize=[] # stores maximum sketch size for each running for l
    avsize=[] # stores average sketch size for each running for l
    sketchtime=[] # stores running time for each running for l
    for f in range(repeattime):
        priority={} # initiate priority queue to store rows
        err=[] # stores err at each measure point
        sketchsize=[] # stores sketch size at each measure point
        L=1 # num_level
        ll=10 # l in LH method to approximate window norm
        block={} #stored blocks where each block contains information of start time, end time, norm of block
        block['level1']=[]
        underblock=[0,0,0] # initiate a block under construction
        for i in range(1,l+1):
            priority['copyX'.replace('X',str(i))]=[] # initiate l priority queue
        start=time.time() # start time to run algorithm
        for k in range(1,n+1):
            if underblock[0]==0:
                underblock[0]=k # start time of tis block
            underblock[1]=k # end time of this block
            underblock[2]+=linalg.norm(A[k-1,:m])**2 # norm of this block
            if underblock[2]>ll*mi: # if norm of under_construction block is bigger than ll*mi
                block['level1'].append(underblock) # put under_construction block into block structure
                underblock=[0,0,0] # initiate a new block under construction
                for i in range(1,L+1):
                    if len(block['levelX'.replace('X',str(i))])>ll: # if num of block in level i is bigger than ll, pop out most old two blocks
                        block1=block['levelX'.replace('X',str(i))].pop(0) 
                        block2=block['levelX'.replace('X',str(i))].pop(0)
                        blockmerged=normmerge(block1,block2) # merge two blocks
                        if i==L:
                            L+=1
                            block['levelX'.replace('X',str(L))]=[]
                        block['levelX'.replace('X',str(i+1))].append(blockmerged) # put the merged block in level i+1
            if k>N:
                normremove(k,N,L) # only start to remove block when rows passed is more than a window size predefined
            wi=linalg.norm(A[k-1,:m])**2 # norm of row k
            if wi==0: # if row k is all zero
                continue
            ui=random.uniform(0,1,l) # generate l uniform distributed random number
            pi=ui**(1/wi) # set p, row k have each pi for each level i
            for i in range(1,l+1):# for each level i from 1 to l
                if len(priority['copyX'.replace('X',str(i))])==0:
                    priority['copyX'.replace('X',str(i))].append((k,pi[i-1],A[k-1].reshape(1,m))) # start time, p, row
                else:
                    for item in priority['copyX'.replace('X',str(i))]:
                        if k-item[0]>N or item[1]<pi[i-1]: # remove rows that are out of window or have a p smaller than p of row k in this level i
                            priority['copyX'.replace('X',str(i))].remove(item)
                    priority['copyX'.replace('X',str(i))].append((k,pi[i-1],A[k-1].reshape(1,m))) # append row k to level i
            if k>=N and np.remainder(k,jg)==0: # measure err between window and sketch
                B=max(priority['copy1'],key=lambda x: x[1])[2] #sketch
                num=len(priority['copy1']) # number of rows stored
                for i in range(2,l+1):
                    B=np.append(B,max(priority['copyX'.replace('X',str(i))],key=lambda x: x[1])[2],axis=0)
                    num=num+len(priority['copyX'.replace('X',str(i))])
                window_norm=calnorm() # norm of current window
                for i in range(l):
                    B[i]=B[i]*np.sqrt(window_norm/l)/linalg.norm(B[i]) # adjust norm of sketch
                err.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(B.T,B),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
                sketchsize.append(num) # append calculated sketch size
        sketchtime.append(time.time()-start) # count algorithm running time
        maxerr.append(np.max(err)) # maximum err
        averr.append(np.mean(err)) # average err
        maxsize.append(np.max(sketchsize)) # maximum sketch size
        avsize.append(np.mean(sketchsize)) # average sketch size
    nswtime.append(np.mean(sketchtime)) # average for all run
    nswmaxerr.append(np.mean(maxerr)) # average for all run
    nswaverr.append(np.mean(averr)) # average for all run
    nswmaxsketchsize.append(np.mean(maxsize)) # average for all run
    nswavsketchsize.append(np.mean(avsize)) # average for all run

# save results for norm sampling with replacement
np.savetxt("samplingwl.txt",sl1)
np.savetxt('NSWmaxerr.txt',nswmaxerr)
np.savetxt('NSWaverr.txt',nswaverr)
np.savetxt('NSWmaxsketchsize.txt',nswmaxsketchsize)
np.savetxt('NSWavsketchsize.txt',nswavsketchsize)
np.savetxt('NSWtime.txt',nswtime)

sl2=[3,5,8,10,15,20,25,30,35,40,50,65] # l run for this algorithm
nsnmaxerr=[] # stores maximum err for each l for SWOR
nsnaverr=[] # stores average err for each l for SWOR
nsnmaxsketchsize=[] # stores maximum sketch size for each l for SWOR
nsnavsketchsize=[] # stores average sketch size for each l for SWOR
nsntime=[] # stores running time for each l for SWOR

tmaxerr=[] # stores maximum err for each l for SWOR-ALL
taverr=[] # stores average err for each l for SWOR-ALL
#Norm Sampling without replacement
for l in sl2:
    maxerr=[] # stores maximum err for each running for l for SWOR
    maxerr2=[] # stores maximum err for each running for l for SWOR-ALL
    averr=[] # stores average err for each running for l for SWOR
    averr2=[] # stores average err for each running for l for SWOR-ALL
    maxsize=[] # stores maximum sketch size for each running for l, same with  SWOR and SWOR-ALL
    avsize=[] # stores average sketch size for each running for l, same with  SWOR and SWOR-ALL
    sketchtime=[] # stores running time for each running for l, same with  SWOR and SWOR-ALL
    for f in range(repeattime):
        err=[] # stores err at each measure point for SWOR
        err2=[] # stores err at each measure point for SWOR-ALL
        priority=[] # initiate priority queue to store rows
        sketchsize=[] # stores sketch size at each measure point
        L=1 # num_level
        ll=10 # l in LH method to approximate window norm
        block={} #stored blocks where each block contains information of start time, end time, norm of block
        block['level1']=[]
        underblock=[0,0,0] # initiate a block under construction
        start=time.time() # start time to run algorithm
        for k in range(1,n+1):
            if underblock[0]==0:
                underblock[0]=k # start time of tis block
            underblock[1]=k # end time of this block
            underblock[2]+=linalg.norm(A[k-1,:m])**2 # norm of this block
            if underblock[2]>ll*mi: # if norm of under_construction block is bigger than ll*mi
                block['level1'].append(underblock) # put under_construction block into block structure
                underblock=[0,0,0] # initiate a new block under construction
                for i in range(1,L+1):
                    if len(block['levelX'.replace('X',str(i))])>ll: # if num of block in level i is bigger than ll, pop out most old two blocks
                        block1=block['levelX'.replace('X',str(i))].pop(0)
                        block2=block['levelX'.replace('X',str(i))].pop(0)
                        blockmerged=normmerge(block1,block2) # merge two blocks
                        if i==L:
                            L+=1
                            block['levelX'.replace('X',str(L))]=[]
                        block['levelX'.replace('X',str(i+1))].append(blockmerged) # put the merged block in level i+1
            if k>N:
                normremove(k,N,L)  # only start to remove block when rows passed is more than a window size predefined
            wi=linalg.norm(A[k-1,:m],2)**2 # norm of row k
            if wi==0: # if row k is all zero
                continue
            ui=random.uniform(0,1,1) # generate 1 uniform distributed random number
            pi=ui**(1/wi) # set p
            if len(priority)==0:
                priority.append([k,pi,A[k-1].reshape(1,m),0]) # start time, p, row, number of rows in the queue that have smaller p than row k 
            else:
                for item in priority:
                    if k-item[0]>N:
                        priority.remove(item) # remove rows that are out of window
                    else:
                        if item[1]<pi:
                            item[3]+=1 # ki increase by 1
                            if item[3]>l: #  if ki is bigger than l
                                priority.remove(item) #  remove it
                priority.append([k,pi,A[k-1].reshape(1,m),0]) # append row k in the queue
            if k>=N and np.remainder(k,jg)==0: # measure err between window and sketch
                priority=sorted(priority,key=lambda x: x[1],reverse=True)
                B=priority[0][2]
                for i in range(1,l):# using l rows
                    B=np.append(B,priority[i][2],axis=0)
                window_norm=calnorm()  # norm of current window
                for i in range(l):
                    B[i]=B[i]*np.sqrt(window_norm/l)/linalg.norm(B[i]) # adjust norm of SWOR sketch
                err.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(B.T,B),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
                BB=priority[0][2]
                length=len(priority) # number of rows stored
                for i in range(1,length):# using all rows
                    BB=np.append(BB,priority[i][2],axis=0)
                for i in range(length):
                    BB[i]=BB[i]*np.sqrt(window_norm/length)/linalg.norm(BB[i]) # adjust norm of SWOR-ALL sketch
                err2.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(BB.T,BB),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
                sketchsize.append(len(priority)) # append calculated sketch size
        sketchtime.append(time.time()-start) # count algorithm running time
        maxerr.append(np.max(err)) # maximum err for SWOR
        averr.append(np.mean(err)) # average err for SWOR
        maxerr2.append(np.max(err2)) # maximum err for SWOR-ALL
        averr2.append(np.mean(err2)) # average err for SWOR-ALL
        maxsize.append(np.max(sketchsize)) # maximum sketch size
        avsize.append(np.mean(sketchsize)) # average sketch size
    nsntime.append(np.mean(sketchtime))  # average for all run
    nsnmaxerr.append(np.mean(maxerr))  # average for all run
    nsnaverr.append(np.mean(averr))  # average for all run
    tmaxerr.append(np.mean(maxerr2))  # average for all run
    taverr.append(np.mean(averr2))  # average for all run
    nsnmaxsketchsize.append(np.mean(maxsize))  # average for all run
    nsnavsketchsize.append(np.mean(avsize))  # average for all run

np.savetxt("samplingnl.txt",sl2)
np.savetxt('NSNmaxerr.txt',nsnmaxerr)
np.savetxt('NSNaverr.txt',nsnaverr)
np.savetxt('NSNmaxsketchsize.txt',nsnmaxsketchsize)
np.savetxt('NSNavsketchsize.txt',nsnavsketchsize)
np.savetxt('NSNtime.txt',nsntime)
#Norm Sampling without replacement using all
np.savetxt('TESTmaxerr.txt',tmaxerr)
np.savetxt('TESTaverr.txt',taverr)
np.savetxt('TESTmaxsketchsize.txt',nsnmaxsketchsize)
np.savetxt('TESTavsketchsize.txt',nsnavsketchsize)
np.savetxt('TESTtime.txt',nsntime)

#func used in Logarithmic method
def mergefd(block1,block2): # merge two blocks using Frequent-Direction algorithm
    global l # accecss to global variable, current l
    blockmerged=[0,0,[],0] # initiate a new block
    blockmerged[0]=block1[0]# the "older" block's start time
    blockmerged[1]=block2[1]# the "newer" block's end time
    A=np.append(block1[2],block2[2],axis=0) # combine two blocks' rows
    U,S,V=linalg.svd(A,full_matrices=0) # conduct Singular Value Decomposition
    if l<len(S): # reduce the last singular value
        SS=S**2-S[l]**2
    else:
        SS=S**2-S[-1]**2
    SS[SS<0]=0
    B=dot(sqrt(diag(SS[:l])),V[:l,:]) # get the svd matrix
    blockmerged[2]=B # save the svd matrix
    blockmerged[3]=block1[3]+block2[3] # new block's norm eaquals to the sum of two blocks' norm
    return blockmerged

def eremove(k,N,level): # remove rows in level that expires in the window
    global L # current num_level
    global block
    if level!=0:
        for item in block['levelX'.replace('X',str(level))]:  # for every row in level
            if k-item[0]>N: # if row's start time is out of current window
                block['levelX'.replace('X',str(level))].remove(item)
            else:
                break # if this row's start time is not out of current window, sure later rows are not neither
        if len(block['levelX'.replace('X',str(level))])==0 and level!=1: #if rows in this level is all removed
            L=L-1 # current num_level should minus 1
            eremove(k,N,level-1) # continur to remove rows in level-1 if any expires.

el=range(2,8,1) # l run for this algorithm
efdmaxerr=[] # stores maximum err for each l
efdaverr=[] # stores average err for each l
efdmaxsketchsize=[] # stores maximum sketch size for each l
efdavsketchsize=[] # stores average sketch size for each l
efdtime=[] # stores running time for each l

#LM Frequent Direction
for l in el:
    err=[]  # stores err at each measure point for LM-FD
    sketchsize=[] # stores sketch size at each measure point
    ebzn=1/l # 
    L=1 # num_level
    block={} # initiate a LM structure
    block['level1']=[] #stored blocks where each block contains information of start time, end time, rows information, norm of block
    underblock=[0,0,[],0]# initiate a block under construction
    start=time.clock()# start time to run algorithm
    for k in range(1,n+1):
        if underblock[0]==0:
            underblock[0]=k # start time of tis block
        underblock[1]=k # end time of this block
        underblock[2].append(A[k-1].reshape(m)) # stored raw rows
        underblock[3]+=linalg.norm(A[k-1])**2 # norm of this block
        if underblock[3]>l*mi: # if norm of under_construction block is bigger than l*mi
            underblock[2]=np.array(underblock[2])
            block['level1'].append(underblock) # put under_construction block into block structure
            underblock=[0,0,[],0] # initiate a new block under construction
            for i in range(1,L+1): 
                if len(block['levelX'.replace('X',str(i))])>l: # if num of block in level i is bigger than l, pop out most old two blocks
                    block1=block['levelX'.replace('X',str(i))].pop(0)
                    block2=block['levelX'.replace('X',str(i))].pop(0)
                    blockmerged=mergefd(block1,block2)# merge two blocks
                    if i==L:
                        L+=1
                        block['levelX'.replace('X',str(L))]=[]
                    block['levelX'.replace('X',str(i+1))].append(blockmerged)# put the merged block in level i+1
        if k>N: 
            eremove(k,N,L) # only start to remove block when rows passed is more than a window size predefined
        if k>=N and np.remainder(k,jg)==0: # measure err between window and sketch
            answerlist=[]
            for i in range(L,0,-1):
                if len(block['levelX'.replace('X',str(i))])!=0:
                    for item in block['levelX'.replace('X',str(i))]:
                        answerlist.append(item[2])
            if len(underblock[2])!=0:
                answerlist.append(np.array(underblock[2]))
            B=answerlist[0]
            for item in answerlist[1:]:
                B=np.append(B,item,axis=0)
            err.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(B.T,B),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
            sketchsize.append(B.shape[0]) # append sketch size
    efdtime.append(time.clock()-start) # count algorithm running time
    efdmaxerr.append(np.max(err)) # maximum err for LM-FD
    efdaverr.append(np.mean(err)) # average err for LM-FD
    efdmaxsketchsize.append(np.max(sketchsize)) # maximum sketch size
    efdavsketchsize.append(np.mean(sketchsize)) # average sketch size
np.savetxt("EHl.txt",el)
np.savetxt('EHFDmaxerr.txt',efdmaxerr)
np.savetxt('EHFDaverr.txt',efdaverr)
np.savetxt('EHFDmaxsketchsize.txt',efdmaxsketchsize)
np.savetxt('EHFDavsketchsize.txt',efdavsketchsize)
np.savetxt('EHFDtime.txt',efdtime)

#func used in dyadic
def cal(activeblock, sketchsize, L): # calculate the space used in Dyadic Interval framwork
    num=0
    for i in range(1,L+1):
        if len(activeblock['levelX'.replace('X',str(i))])!=0:
            num=num+len(activeblock['levelX'.replace('X',str(i))])*sketchsize[i-1]
    return num

def findblock(level,s,t): # find blocks that are used to construct the sketch, see the paper for the rule
    global answerlist, activeblock
    if level!=0 and s<t:
        if len(activeblock['levelX'.replace('X',str(level))])==0:
            findblock(level-1,s,t)
        else:
            ss=activeblock['levelX'.replace('X',str(level))][-1][1]
            tt=activeblock['levelX'.replace('X',str(level))][0][0]
            findblock(level-1,s,tt)
            if len(activeblock['levelX'.replace('X',str(level))])<=2 and activeblock['levelX'.replace('X',str(level))][0][0]>=s and activeblock['levelX'.replace('X',str(level))][-1][1]<=t:
                for item in activeblock['levelX'.replace('X',str(level))]:
                    answerlist.append(item[3])
            else:
                if activeblock['levelX'.replace('X',str(level))][0][0]>=s and activeblock['levelX'.replace('X',str(level))][0][1]<=t:
                    answerlist.append(activeblock['levelX'.replace('X',str(level))][0][3])
                else:
                    if activeblock['levelX'.replace('X',str(level))][-1][0]>=s and activeblock['levelX'.replace('X',str(level))][-1][1]<=t:
                        answerlist.append(activeblock['levelX'.replace('X',str(level))][-1][3])
            findblock(level-1,ss,t)

def combinefd(answerlist,l):# combine blocks and conduct the Frequent-Direction Algo to get the sketch
    B=answerlist[0]
    for item in answerlist[1:]:
        B=np.append(B,item,axis=0)
    U,S,V=linalg.svd(B,full_matrices=0)
    if l<len(S):
        SS=S**2-S[l]**2
    else:
        SS=S**2-S[-1]**2
    SS[SS<0]=0
    BB=dot(sqrt(diag(SS[:l])),V[:l,:])
    return BB

def dremove(): # remove blocks that are expired
    global activeblock, k, L, N
    for i in range(1,L+1):
        for item in activeblock['levelX'.replace('X',str(i))]:
            if item[0]<k-N:
                activeblock['levelX'.replace('X',str(i))].remove(item)
                break

def trazero(k): # get the number of trailing zero for input number k
    num=0
    string=np.binary_repr(k)
    for i in range(len(string)-1,-1,-1):
        if string[i]=='0':
            num+=1
        else:
            break
    return num

def dmerge(underblock,bucket,l): # merge block under construction and bucket
    if underblock.max()==0 and underblock.min()==0:
        A=bucket
        return A
    else:
        A=np.append(underblock,bucket,axis=0)
        U,S,V=linalg.svd(A,full_matrices=0)
        if l<len(S):
            SS=S**2-S[l]**2
        else:
            SS=S**2-S[-1]**2
        SS[SS<0]=0
        B=dot(sqrt(diag(SS[:l])),V[:l,:])
        return B

#dyadic
dl=range(5,45,5) # l run for this algorithm
#Frequent Direction
dfdtime=[]  # stores running time for each l
dfdmaxerr=[]# stores maximum err for each l
dfdaverr=[] # stores average err for each l
dfdmaxsketchsize=[] # stores maximum sketch size for each l
dfdavsketchsize=[] # stores average sketch size for each l

for l in el:
    err=[]  # stores err at each measure point for LM-FD
    sketchsize=[] # stores sketch size at each measure point
    ebzn=1/l # 
    L=1 # num_level
    block={} # initiate a LM structure
    block['level1']=[] #stored blocks where each block contains information of start time, end time, rows information, norm of block
    underblock=[0,0,[],0]# initiate a block under construction
    start=time.clock()# start time to run algorithm
    for k in range(1,n+1):
        if underblock[0]==0:
            underblock[0]=k # start time of tis block
        underblock[1]=k # end time of this block
        underblock[2].append(A[k-1].reshape(m)) # stored raw rows
        underblock[3]+=linalg.norm(A[k-1])**2 # norm of this block
        if underblock[3]>l*mi: # if norm of under_construction block is bigger than l*mi
            underblock[2]=np.array(underblock[2])
            block['level1'].append(underblock) # put under_construction block into block structure
            underblock=[0,0,[],0] # initiate a new block under construction
            for i in range(1,L+1): 
                if len(block['levelX'.replace('X',str(i))])>l: # if num of block in level i is bigger than l, pop out most old two blocks
                    block1=block['levelX'.replace('X',str(i))].pop(0)
                    block2=block['levelX'.replace('X',str(i))].pop(0)
                    blockmerged=mergefd(block1,block2)# merge two blocks
                    if i==L:
                        L+=1
                        block['levelX'.replace('X',str(L))]=[]
                    block['levelX'.replace('X',str(i+1))].append(blockmerged)# put the merged block in level i+1
        if k>N: 
            eremove(k,N,L) # only start to remove block when rows passed is more than a window size predefined
        if k>=N and np.remainder(k,jg)==0: # measure err between window and sketch
            answerlist=[]
            for i in range(L,0,-1):
                if len(block['levelX'.replace('X',str(i))])!=0:
                    for item in block['levelX'.replace('X',str(i))]:
                        answerlist.append(item[2])
            if len(underblock[2])!=0:
                answerlist.append(np.array(underblock[2]))
            B=answerlist[0]
            for item in answerlist[1:]:
                B=np.append(B,item,axis=0)
            err.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(B.T,B),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
            sketchsize.append(B.shape[0]) # append sketch size
    efdtime.append(time.clock()-start) # count algorithm running time
    efdmaxerr.append(np.max(err)) # maximum err for LM-FD
    efdaverr.append(np.mean(err)) # average err for LM-FD
    efdmaxsketchsize.append(np.max(sketchsize)) # maximum sketch size
    efdavsketchsize.append(np.mean(sketchsize)) # average sketch size
for l in dl:
    err=[] # stores err at each measure point for DI-FD
    sketchnum=[] # stores sketch size at each measure point
    L=int(ceil(log2(l))) # num_level
    blocksize=[]
    small=floor(mi*N/l)
    for i in range(1,L+1):
        blocksize.append(small*2**(i-1))
    sketchsize=[]
    for i in range(1,L+1):
        sketchsize.append(2**(i))
    activeblock={}
    underblock={}
    for i in range(1,L+1):# initiate the activeblock and underblock structure
        activeblock['levelX'.replace('X',str(i))]=[]
        underblock['levelX'.replace('X',str(i))]=[1,0,0,zeros(sketchsize[i-1]*m).reshape(sketchsize[i-1],m)]
    length=0
    bucket=[]
    start=time.clock()
    for k in range(1,n+1):
        if len(bucket)==0: # if the bucket is empty
            bucket=A[k-1].reshape([1,m]) #put the row into the bucket
        else:
            bucket=np.append(bucket,A[k-1].reshape([1,m]),axis=0)# else append the row into the bucket
        if bucket.shape[0]==l: # bucket is full
            for i in range(1,L+1): # update block under construction on each level
                underblock['levelX'.replace('X',str(i))][1]=k #update end time
                underblock['levelX'.replace('X',str(i))][2]+=linalg.norm(bucket)**2 # update norm
                underblock['levelX'.replace('X',str(i))][3]=dmerge(underblock['levelX'.replace('X',str(i))][3],bucket,sketchsize[i-1]) # update rows information
            bucket=[] # initiate a new bucket
        if underblock['level1'][2]>=small: # block under construction is full
            s=trazero(length+1)+1
            if s>L:
                s=L
            length+=1
            for i in range(1,s+1):
                activeblock['levelX'.replace('X',str(i))].append(underblock['levelX'.replace('X',str(i))]) # append the block under construction to the frame in level i
                underblock['levelX'.replace('X',str(i))]=[k+1,0,0,zeros(sketchsize[i-1]*m).reshape(sketchsize[i-1],m)] # initiate a new block under construction in level i
        if k>N:
            dremove() # remove blocks that are expired
        if k>=N and np.remainder(k,jg)==0: # measure err between window and sketch
            sketchnum.append(cal(activeblock,sketchsize,L)) # space used to construct the DI framwork
            answerlist=[]
            findblock(L,k-N,k+1)# find blocks that should be used to construct the sketch
            if underblock['level1'][3].max()!=0:
                answerlist.append(underblock['level1'][3]) # append the block under construction if it is not empty
            if len(bucket)>0:
                answerlist.append(bucket) # append the bucket if it is not empty
            BB=combinefd(answerlist,l)
            err.append(linalg.norm(dot(A[k-N:k,:m].T,A[k-N:k,:m])-dot(BB.T,BB),2)/linalg.norm(A[k-N:k,:m])**2) # append calculated err
    dfdtime.append(time.clock()-start) # count algorithm running time
    dfdmaxerr.append(np.max(err)) # maximum err for DI-FD
    dfdaverr.append(np.mean(err)) # average err for LM-FD
    dfdmaxsketchsize.append(np.max(sketchnum)) # maximum sketch size
    dfdavsketchsize.append(np.mean(sketchnum)) # average sketch size

np.savetxt("dyadicl.txt",dl)
np.savetxt('dFDmaxerr.txt',dfdmaxerr)
np.savetxt('dFDaverr.txt',dfdaverr)
np.savetxt('dFDmaxsketchsize.txt',dfdmaxsketchsize)
np.savetxt('dFDavsketchsize.txt',dfdavsketchsize)
np.savetxt('dFDtime.txt',dfdtime)