#pragma once
#include <mpi.h>
#include <math.h>	
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <vector>
#include <complex>

using namespace std;

namespace advection
{
	class advection_2d
	{
	protected:
		int solver_type;
		int size;
		int rank;
		double * face_normals;
		double * tr_areas;
		double * xbarys;
		double * ybarys;
		int * neighbor_ids;
		double limiter(double const& r_factor, double const& weight);
		void solver_small_cpu(double *&u, double const &dt, double *&velocities, double *&interpolated_velocities, bool if_y, bool if_h, double t);

	public:
		bool * is_boundary;
		void calc_dynamic_arrays();
		advection_2d(int solver_type, int size, int rank) : solver_type(solver_type), size(size), rank(rank)
		{
			calc_dynamic_arrays();
		}
		~advection_2d();
		virtual void solver_small(double *&u, double const dt, double *&coefs, bool if_y, bool if_h, double t);
	};

	class advection_2d_mpi : public advection_2d
	{
	protected:
		int size_of_group;
		int * interfaces_size;
		int * solver_types;
		int ** received_ids;

		double ** send_data;
		double ** receive_data;
		int ** ids_to_send;

		void calc_mpi_arrays();
	public:
		int * global_ids_reference;
		advection_2d_mpi(int solver_type, int rank, int size_of_group, int size, int * solver_types) : advection_2d(solver_type, size, rank), size_of_group(size_of_group), solver_types(solver_types)
		{
			calc_mpi_arrays();
		}
		~advection_2d_mpi();
		virtual void solver_small(double *& u, double const& dt, double*& coefs, bool if_y, bool if_h, double t);
	};

}
