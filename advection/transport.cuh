#pragma once
#include <mpi.h>
#include <math.h>	
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <complex>

namespace advection
{
    void initialize_arrays(int solver_type, int size, bool *& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys);
    void dealloc_arrays(int solver_type, bool *& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys);
    void call_solver_small_cuda(double *&u, double const dt, double *&velocities, bool * is_boundary, double * face_normals, int * neighbor_ids, double * tr_areas, double * xbarys, double * ybarys, int size, bool if_y, bool if_h, double t);
    void call_set_send(double *&u, double *& velocities, double *& send_data, int *& ids_to_send, const int & interfaces_size);
    void call_set_receive(double *&u, double *& velocities, double *& receive_data, int *& received_ids, const int & interfaces_size);
    void call_alloc_transfer_array(double *& send_data, double *& receive_data, int size);
    void call_free_transfer_array(double *& send_data, double *& receive_data);

    //template<class T>	
    void call_cuda_malloc(int*& array, const int & size);

    //template<class T>
    void call_cuda_free(int*& array);
}
