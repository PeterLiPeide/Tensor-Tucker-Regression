"""
Test File
Tucker regression
"""
#%%
import numpy as np 
import tensorly as tl 
import Tucker_ver1 as TKR 
import scipy.io as sio
import os

#%%
# Data generation, import directly from matlab
dataX_file = 'D:\\OneDrive - Michigan State University\\Documents\\Tensor Regression\\Tensor_Tucker\\Code\\X_Simu'
dataY_file = 'D:\\OneDrive - Michigan State University\\Documents\\Tensor Regression\\Tensor_Tucker\\Code\\Y_Simu'
listX = os.listdir(dataX_file)
listY = os.listdir(dataY_file)
Ty = [np.array(sio.loadmat(os.path.join(dataX_file, listX[0]))['X'])]
Tx = [np.array(sio.loadmat(os.path.join(dataY_file, listY[0]))['Y'])]



# %%
mymodel = TKR.Tucker_Binary_Regression()
mymodel.fit(X=Tx, Y=Ty, lambda_regula=0.5)

# %%
mymodel.Train_plot()

#%%
newY = [np.array(sio.loadmat(os.path.join(dataX_file, listX[1]))['X']), \
    np.array(sio.loadmat(os.path.join(dataX_file, listX[2]))['X'])]
newX = [np.array(sio.loadmat(os.path.join(dataY_file, listY[1]))['Y']), \
    np.array(sio.loadmat(os.path.join(dataY_file, listY[2]))['Y'])]

# print(mymodel.predict(newX=newY, newY=newX))

# %%
