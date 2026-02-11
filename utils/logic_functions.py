import numpy as np

def logical_and(x):
    O = np.logical_and(x[:,0], x[:,1])
    O = np.reshape(O, (x.shape[0],1))
    return O.astype('int8')


def xor(x):
    O1 = np.logical_xor(x[:,0],x[:,1])
    O1 = np.reshape(O1,(x.shape[0],1))
    return O1.astype('int8')

def full_adder(x):
    O1 = np.logical_xor(np.logical_xor(x[:,0], x[:,2]), x[:,1])
    O2 = np.logical_or( np.logical_and(x[:,0],x[:,2]), np.logical_and(x[:,1], np.logical_xor(x[:,0],x[:,2])))
    #print(x[:,0], x[:,1], x[:,2], O2, O1)
    O2 = np.reshape(O2, (x.shape[0], 1))
    O1 = np.reshape(O1, (x.shape[0],1))
    res = np.concatenate((O1,O2), axis = 1)
    #print(res)
    return (res).astype('int8')

def full_adder_4bit(x):
    UE1 = np.zeros_like(x[:,0])
    out = []
    for i in range(0,4):
        I1 = np.reshape(x[:,i], ( x.shape[0],1))
        I2 = np.reshape(x[:,i+4], (x.shape[0],1))
        UE1 = np.reshape( UE1, (UE1.shape[0], 1))
        #print('Input')
        #print(I1)
        #print(I2)
        #print(UE1)
        O = full_adder(np.concatenate((I1,I2,UE1), axis = 1))
        #print('O')
        #print(O)
        out.append(np.reshape(O[:,0], (O[:,0].shape[0],1)))
        UE1 = O[:,1]
        #input()

    #print(out)
    res = np.concatenate(out, axis = 1)
    return (res).astype('int8')

def full_adder_nbit(x,n):
    UE1 = np.zeros_like(x[:,0])
    out = []
    for i in range(0,n):
        I1 = np.reshape(x[:,i], ( x.shape[0],1))
        I2 = np.reshape(x[:,i+n], (x.shape[0],1))
        UE1 = np.reshape( UE1, (UE1.shape[0], 1))
        #print('Input')
        #print(I1)
        #print(I2)
        #print(UE1)
        O = full_adder(np.concatenate((I1,I2,UE1), axis = 1))
        #print('O')
        #print(O)
        out.append(np.reshape(O[:,0], (O[:,0].shape[0],1)))
        UE1 = O[:,1]
        #input()

    #print(out)
    res = np.concatenate(out, axis = 1)
    return (res).astype('int8')


def SR_latch(x):
    O1 = np.logical_not(np.logical_or(x[:,0], x[:,2]))
    O2 = np.logical_not(np.logical_or(x[:,1], x[:,3]))

    O2 = np.reshape(O2, (x.shape[0], 1))
    O1 = np.reshape(O1, (x.shape[0],1))
    res = np.concatenate((O1,O2),axis = 1)
    #res = O1
    return (res).astype('int8')



def SR_latch_multi(x):
    num = x.shape[1]//2
    out = []
    for i in range(0,num):
        I1 = np.reshape(x[:,i], (x.shape[0],1))
        I2 = np.reshape(x[:,i+1],(x.shape[0],1))
        O = SR_latch(np.concatenate((I1,I2), axis = 1))
        out.append(np.reshape(O[:,0], (O[:,0].shape[0],1)))

    res = np.concatenate(out, axis = 1)
    return (res).astype('int8')




def multiple_and_function(x):
    O = np.zeros((x.shape[0],0))
    for i in range(0,4):
        O1 = np.logical_and( x[:,0+i], x[:,4+i])
        O1 = np.reshape(O1, (x.shape[0],1))
        O = np.concatenate((O,O1),axis = 1)
        #O.append(O1)
    return (O).astype('int8')



