# 2D Ising model simulation 
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAising2D.cu, animation2D.py, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/animation2D
#                               https://github.com/bvillasen/tools
import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
animation2DDirectory = parentDirectory + "/anim2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )

import animation2D
from cudaTools import setCudaDevice, getFreeMemory, gpuArray2DtocudaArray

nPoints = 512*2
useDevice = None
for option in sys.argv:
  #if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nData = nWidth*nHeight

temp = 1.
beta = np.float32( 1./temp)

#Convert parameters to float32



#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 16   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
grid2D = (gridx, gridy, 1)
block2D = (block_size_x, block_size_y, 1)


#Initialize openGL
animation2D.nWidth = nWidth
animation2D.nHeight = nHeight
animation2D.initGL()

#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "Compiling CUDA code"
cudaCodeString_raw = open("CUDAising2D.cu", "r").read()
cudaCodeString = cudaCodeString_raw # % { "BLOCK_WIDTH":block2D[0], "BLOCK_HEIGHT":block2D[1], "BLOCK_DEPTH":block2D[2], }
cudaCode = SourceModule(cudaCodeString)
tex_spins = cudaCode.get_texref('tex_spinsIn')
isingKernel = cudaCode.get_function('ising_kernel')
########################################################################
def sendToScreen( plotData ):
  #maxVal = gpuarray.max(plotData).get() + 0.00005
  #multiplyByFloat( 1./maxVal, plotData )
  floatToUchar( plotData, plotData_d )
  copyToScreenArray()
########################################################################
def swipe():
  randomNumbers_d = curandom.rand((nData))
  stepNumber = np.int32(0)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), beta, 
	       spinsOut_d, randomNumbers_d, grid=grid2D, block=block2D )
  copy2D_dtod(aligned=True) 

  stepNumber = np.int32(1)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), beta,
	       spinsOut_d, randomNumbers_d, grid=grid2D, block=block2D )
  copy2D_dtod(aligned=True)
########################################################################

#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
#Set initial random distribution
spins_h = (2*np.random.random_integers(0,1,[nHeight, nWidth]) - 1 ).astype(np.int32)
spinsOut_d = gpuarray.to_gpu( spins_h )
randomNumbers_d = curandom.rand((nData))
#For texture version
spinsInArray_d, copy2D_dtod = gpuArray2DtocudaArray( spinsOut_d )
##For shared version
##memory for plotting
#plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
#volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 



def stepFunction():

  [swipe() for i in range(1)]

  
#configure animation2D stepFunction and plotData
animation2D.stepFunc = stepFunction
animation2D.plotData_d = spinsOut_d
animation2D.maxVar = np.float32(2)
animation2D.minVar = np.float32(-20)

#run volumeRender animation
volumeRender.animate()


