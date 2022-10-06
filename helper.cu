#include "helper.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void initialize_cuda_memory(double * &E_x, double * &E_y, double * &H_z, double * &coefs_E_x, double * &coefs_E_y, double * &coefs_H_z, double * &mu, double * &epsilon, int size)
{
    cudaMallocManaged((void **) &E_y, size*sizeof(double));
    cudaMallocManaged((void **) &E_x, size*sizeof(double));
    cudaMallocManaged((void **) &H_z, size*sizeof(double));
    cudaMallocManaged((void **) &coefs_E_x, size*2*sizeof(double));
    cudaMallocManaged((void **) &coefs_E_y, size*2*sizeof(double));
    cudaMallocManaged((void **) &coefs_H_z, size*2*sizeof(double));
    //maybe leave the next two as cudaMallocManaged
    cudaMallocManaged((void **) &mu, size*sizeof(double));
    cudaMallocManaged((void **) &epsilon, size*sizeof(double));
}

void set_device(int rank)
{
    gpuErrchk(cudaSetDevice(rank));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank);
    printf("Device Number: %d\n", rank);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void call_update_coefs_cuda(double * H_z, double * E_x, double * E_y, double *coefs_H_z,
                            double *coefs_E_x, double *coefs_E_y, double *mu, 
                            double *epsilon, double size)
{
    dim3 block(32);
    dim3 grid_big((size+block.x-1)/block.x);
    update_coefs_cuda<<<grid_big, block>>>(H_z, E_x, E_y, coefs_H_z, coefs_E_x, coefs_E_y, mu, epsilon, size);
    cudaDeviceSynchronize();
}

__global__ void update_coefs_cuda(double * H_z, double * E_x, double * E_y, double *coefs_H_z,
                                  double *coefs_E_x, double *coefs_E_y, double *mu,
                                  double *epsilon, double size)
{
    int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int global_id = (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x
                    + (threadIdx.x + blockIdx.x * blockDim.x);
    for (int p = global_id; p < size; p+=numThreads)
    {
        coefs_H_z[2*p] = 1.0/mu[p]*E_y[p];
        coefs_H_z[2*p+1] = -1.0/mu[p]*E_x[p];
        coefs_E_x[2*p] = 0.0;
        coefs_E_y[2*p] = 1.0/epsilon[p]*H_z[p];
        coefs_E_x[2*p+1] = -1.0/epsilon[p]*H_z[p];
        coefs_E_y[2*p+1] = 0.0;
    }
}

