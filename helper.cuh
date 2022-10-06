#include <algorithm>
#include <memory>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <complex>
#include <cassert>
#include <utility>
#include <cuda_runtime.h>
#include <ctime>
#include <cstdlib>

void initialize_cuda_memory(double *&E_x, double *& E_y, double *& H_z, double *& coefs_E_x, double *& coefs_E_y, double *& coefs_H_z, double *& mu, double *& epsilon, int size);

void set_device(int rank);

void call_update_coefs_cuda(double * H_z, double * E_x, double * E_y, double *coefs_H_z, double *coefs_E_x, double *coefs_E_y, double *mu, double *epsilon, double size);

__global__ void update_coefs_cuda(double * H_z, double * E_x, double * E_y, double *coefs_H_z, double *coefs_E_x, double *coefs_E_y, double *mu, double *epsilon, double size);


