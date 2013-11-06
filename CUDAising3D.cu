// #include <pycuda-complex.hpp>
// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>


texture< int, cudaTextureType3D, cudaReadModeElementType>tex_spinsIn;
//surface<void, 2> surf_out;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


__device__ int deltaEnergy( int nWidth, int nHeight, int nDepth, int t_i, int t_j, int t_k ){
  int center, right, left, up, down, top, bottom;
  center = tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k);
  up =     tex3D(tex_spinsIn, (float)t_j, (float)t_i+1, (float)t_k);
  down =   tex3D(tex_spinsIn, (float)t_j, (float)t_i-1, (float)t_k);
  right =  tex3D(tex_spinsIn, (float)t_j+1, (float)t_i, (float)t_k);
  left =   tex3D(tex_spinsIn, (float)t_j-1, (float)t_i, (float)t_k);
  top =    tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k+1);
  bottom = tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k-1);
  
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down =  tex3D( tex_spinsIn, (float)t_j, (float)nHeight-1, (float)t_k );
  if (t_i == nHeight-1)   up =    tex3D( tex_spinsIn, (float)t_j, (float)0, (float)t_k );
  if (t_j == 0)           left =  tex3D( tex_spinsIn, (float)nWidth-1, (float)t_i, (float)t_k );
  if (t_j == nWidth-1)    right = tex3D( tex_spinsIn, (float)0, (float)t_i, (float)t_k );
  if (t_k == 0)           left =  tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)nDepth-1 );
  if (t_k == nDepth-1)    right = tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)0 );
  
  return 2*center*(up + down + right + left + top + bottom);
}

// __device__ int getSpinEnergy( int nWidth, int nHeight, int t_i, int t_j ){
//   int center = tex2D( tex_spinsIn, t_j, t_i );
//   int left   = tex2D( tex_spinsIn, t_j-1, t_i );
//   int right  = tex2D( tex_spinsIn, t_j+1, t_i );
//   int up     = tex2D( tex_spinsIn, t_j, t_i-1 );
//   int down   = tex2D( tex_spinsIn, t_j, t_i+1 );
//   
//   //Set PERIODIC boundary conditions
//   if (t_i == 0)           down = tex2D( tex_spinsIn, t_j, nHeight-1 );
//   if (t_i == (nHeight-1))   up = tex2D( tex_spinsIn, t_j, 0 );
//   if (t_j == 0)           left = tex2D( tex_spinsIn, nWidth-1, t_i );
//   if (t_j == (nWidth-1)) right = tex2D( tex_spinsIn, 0, t_i );
//   
//   return -center*(up + down + right + left );
// }

__device__ bool metropolisAccept( int tid, float beta, int deltaE, float *randomNumbers){
  float random = randomNumbers[tid];
  float val = exp(-1*beta*deltaE);

  if (deltaE<=0) return true;
  if (random < val) return true;
  return false;
}

__global__ void ising_kernel( int paridad,  int nWidth, int nHeight, int nDepth, float beta, 
			      int *spinsOut, float *randomNumbers ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  int deltaE = deltaEnergy( nWidth, nHeight, nDepth, t_i, t_j, t_k );
  int currentSpin = tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)t_k );
  if ( (t_i+ t_j + t_k)%2 == paridad ){
    if (metropolisAccept(tid, beta, deltaE, randomNumbers)) spinsOut[tid] = -1*currentSpin; 
    //else spinsOut[tid] = currentSpin;
  }
//   if (saveEnergy) spinsEnergies[tid] = getSpinEnergy( nWidth, nHeight, t_i, t_j );
}
























































