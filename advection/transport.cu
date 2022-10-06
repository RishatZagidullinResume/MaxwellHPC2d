#include "transport.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace advection
{
	__global__ void solver_small_cuda(double *u, double dt, double *velocities, double *interpolated_velocities, bool * is_boundary, int * neighbor_ids, double * face_normals, double * tr_areas, double * ybarys, double * xbarys, int size, bool if_y, bool if_h, double t);
	
	//__global__ void set_send(double *&u, double *& velocities, double *&send_data, double ** ids_to_send, double ** interfaces_size);
	//__global__ void set_receive(double *&u, double *& velocities, double *&receive_data, double ** ids_to_send, double ** interfaces_size);

	__global__ void set_send(double *u, double * velocities, double * send_data, int * ids_to_send, int interfaces_size)
	{
		int numThreads = blockDim.x * gridDim.x;
		int global_id =  threadIdx.x + blockIdx.x * blockDim.x;
		for (int j = global_id; j < interfaces_size; j+=numThreads)
		{
			send_data[3*j] = u[ids_to_send[j]];
			send_data[3*j+1] = velocities[ids_to_send[j]*2];
			send_data[3*j+2] = velocities[ids_to_send[j]*2+1];
		}
	}
	
	__global__ void set_receive(double *u, double * velocities, double * receive_data, int * received_ids, int interfaces_size)
	{
	
		int numThreads = blockDim.x * gridDim.x;
		int global_id =  threadIdx.x + blockIdx.x * blockDim.x;
		for (int j = global_id; j < interfaces_size; j+=numThreads)
		{
			int val = received_ids[j];
			u[val] = receive_data[3*j];
			velocities[val*2] = receive_data[3*j+1];
			velocities[val*2+1] = receive_data[3*j+2];
		}
	}
	

	void call_set_receive(double *&u, double *& velocities, double *& receive_data, int *& received_ids, const int & interfaces_size)
	{
		dim3 block(128);
                dim3 grid((interfaces_size+block.x-1)/block.x);

		//std::cout << "grid_size: " << grid_size << " block size: " << block_size << " inteface_size: " << interfaces_size << std::endl;
		set_receive<<<grid, block>>>(u, velocities, receive_data, received_ids, interfaces_size);
		//you should comment it out
		cudaDeviceSynchronize();
	}

	void call_set_send(double *&u, double *& velocities, double *& send_data, int *& ids_to_send, const int & interfaces_size)
	{
		dim3 block(128);
		dim3 grid((interfaces_size+block.x-1)/block.x);
		set_send<<<grid, block>>>(u, velocities, send_data, ids_to_send, interfaces_size);
		//you should comment it out
		cudaDeviceSynchronize();
	}


	__device__ double sigma_device(double const &r, double const &dr, int const &PML, int const & size)
	{
		if (r>(dr*(size-PML))) return pow((r-(size-PML)*dr)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else if (r < dr*(PML)) return pow((PML*dr-r)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else return 0;
	}
	
	//template<class T>
	void call_cuda_malloc(int*& array, const int & size)
	{
		cudaMallocManaged((void **) &array, size);
	}
	
	//template<class T>
	void call_cuda_free(int*& array)
	{
		cudaFree(array);
	}
	
	void call_alloc_transfer_array(double *& send_data, double *& receive_data, int size)
	{
		cudaMallocManaged((void **) &send_data, size*sizeof(double));
		cudaMallocManaged((void **) &receive_data, size*sizeof(double));
	}
	void call_free_transfer_array(double *& send_data, double *& receive_data)
	{
		cudaFree(send_data);
		cudaFree(receive_data);
	}

	void initialize_arrays(int solver_type, int size, bool*& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys)
	{
		if (solver_type == 0)
		{
			is_boundary = new bool [size];
			neighbor_ids = new int [3*size];
			face_normals = new double [3*2*size];
			tr_areas = new double [size];
			xbarys =  new double [size];
			ybarys = new double [size];
		}
		else
		{
			cudaMallocManaged((void **) &is_boundary, size*sizeof(bool));
			cudaMallocManaged((void **) &neighbor_ids, 3*size*sizeof(int));
			cudaMallocManaged((void **) &face_normals, size*3*2*sizeof(double));
			cudaMallocManaged((void **) &tr_areas, size*sizeof(double));
			cudaMallocManaged((void **) &xbarys, size*sizeof(double));
			cudaMallocManaged((void **) &ybarys, size*sizeof(double));
		}
	}

	void dealloc_arrays(int solver_type, bool*& is_boundary, double *& face_normals, int *& neighbor_ids, double *& tr_areas, double *& xbarys, double *& ybarys)
	{
		if (solver_type == 0)
		{
			delete [] face_normals;
			delete [] neighbor_ids;
			delete [] tr_areas;
			delete [] xbarys;
			delete [] ybarys;
			delete [] is_boundary;
		}
		else
		{
			cudaFree(face_normals);
			cudaFree(neighbor_ids);
			cudaFree(tr_areas);
			cudaFree(is_boundary);
			cudaFree(xbarys);
			cudaFree(ybarys);
			cudaFree(is_boundary);
		}
	}
	void call_solver_small_cuda(double *&u, double const dt, double *&velocities, bool * is_boundary, double * face_normals, int * neighbor_ids, double * tr_areas, double * xbarys, double * ybarys, int size, bool if_y, bool if_h, double t)
	{
		
		double * interpolated_velocities;
		cudaMallocManaged((void **) &interpolated_velocities, size*3*2*sizeof(double));
		dim3 block(32);
		dim3 grid((size+block.x-1)/block.x);
		//std::cout << " " << grid.x << " " << block.x << " " << size << std::endl;
		solver_small_cuda<<<grid, block>>>(u, dt, velocities, interpolated_velocities, is_boundary, neighbor_ids, face_normals, tr_areas, ybarys, xbarys, size, if_y, if_h, t);
		//std::cout << "just left kernel" << std::endl;
		cudaDeviceSynchronize();
		cudaFree(interpolated_velocities);			
	}
	
	__global__ void solver_small_cuda(double *u, double dt, double *velocities, double *interpolated_velocities, bool * is_boundary, int * neighbor_ids, double * face_normals, double * tr_areas, double * ybarys, double * xbarys, int size, bool if_y, bool if_h, double t)
	{
		int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
		int global_id = (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x + (threadIdx.x + blockIdx.x * blockDim.x);
		for (int j = global_id; j < size; j+=numThreads)
		{	
			for (int k = 0; k < 3; k++)
			{
				if (neighbor_ids[j*3+k] != -1)
				{
					if (if_h && if_y)
					{
						if (is_boundary[j]!=is_boundary[neighbor_ids[j*3+k]] && ybarys[j] < 0.9 && ybarys[j] > 0.1 && ybarys[neighbor_ids[j*3+k]] < 0.9 && ybarys[neighbor_ids[j*3+k]] > 0.1)
						{
							interpolated_velocities[2*(j*3+k)] = (is_boundary[j] ? 1.0 : -1.0)*cos(25.0*(xbarys[neighbor_ids[j*3+k]] - 1.5*t))+velocities[2*j]+velocities[2*neighbor_ids[j*3+k]];
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]+velocities[2*neighbor_ids[j*3+k]+1];
						}
						else
						{
							interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]] + velocities[2*j];
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1] + velocities[2*j+1];
						}
					}
					else
					{
						interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]] + velocities[2*j];
						interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1] + velocities[2*j+1];
					}
				}
				else
				{
					interpolated_velocities[2*(j*3+k)] = velocities[2*j]+0.0;
					interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]+0.0;
				}
			}
			double temp = 0.0;
			for (int k = 0; k < 3; k++)
			{
				temp += interpolated_velocities[2*(3*j+k)] * face_normals[2*(j*3+k)] + interpolated_velocities[2*(3*j+k)+1] * face_normals[2*(j*3+k)+1]; 
			}
			if (!if_h) u[j] = u[j] - dt * (temp/ tr_areas[j] + (if_y ? 0.5*pow(2.25, 0.5)*sigma_device(ybarys[j], 0.01, 10, 100)*u[j] : 0.5*pow(2.25, 0.5)*sigma_device(xbarys[j], 0.01, 20, 200)*u[j]))/(1.0+0.5*dt*( if_y ? pow(2.25, 0.5)*sigma_device(ybarys[j], 0.01, 10, 100) : pow(2.25, 0.5)*sigma_device(xbarys[j], 0.01, 20, 200)));
			else u[j] = u[j] - dt * (temp / tr_areas[j] + 0.5*pow(1.0/2.25, 0.5)*sigma_device(ybarys[j], 0.01, 10, 100)*u[j] + 0.5*pow(1.0/2.25, 0.5)*sigma_device(xbarys[j], 0.01, 20, 200)*u[j])/(1.0+0.5*dt*( pow(1.0/2.25, 0.5)*sigma_device(ybarys[j], 0.01, 10, 100) + pow(1.0/2.25, 0.5)*sigma_device(xbarys[j], 0.01, 20, 200)));
		}
	}

	
//mu_1 = 1.0, mu_2 = 1.0, epsilon_1 = 1.0, epsilon_2 = 2.25
}
