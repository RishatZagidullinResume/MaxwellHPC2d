#include "transport.h"
#include "transport.cuh"
namespace advection
{
	double sigma(double const &r, double const &dr, int const &PML, int const & size)
	{
		if (r>(dr*(size-PML))) return pow((r-(size-PML)*dr)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else if (r < dr*(PML)) return pow((PML*dr-r)/(PML*dr),2)*3.0*log(10.0)*13.0/(PML*dr);
		else return 0;
	}

	advection_2d::~advection_2d()
	{
		dealloc_arrays(solver_type, is_boundary, face_normals, neighbor_ids, tr_areas, xbarys, ybarys);
	}

	advection_2d_mpi::~advection_2d_mpi()
	{
                for (int i = 0; i < size_of_group; i++)
		{
			if ((solver_type==1) && (solver_types[i] == 1) && (i != rank))
			{
				call_free_transfer_array(send_data[i], receive_data[i]);
			}
			else
			{
				if (interfaces_size[i]!=0)
				{
					delete [] receive_data [i];
					delete [] send_data [i];
				}
			}
		}
		delete [] send_data;
		delete [] receive_data;
		for (int i = 0; i < size_of_group; i++)
		{
			if ((solver_type==1) && (solver_types[i] == 1) && (i != rank))
			{
				call_cuda_free(ids_to_send[i]);
				call_cuda_free(received_ids[i]);
			}
			else
			{
				delete [] ids_to_send[i];
				delete [] received_ids[i];
			}
		}
		delete [] received_ids;
		delete [] ids_to_send;
		delete [] interfaces_size;
		delete [] global_ids_reference;
		
	}

	double advection_2d::limiter(double const &r_factor, double const &weight)
	{
		return max(0.0, min(weight*r_factor,min(0.5*(1.0+r_factor), weight)));//MC
	}
	

	void advection_2d::calc_dynamic_arrays()
	{
		std::stringstream ss;
		ss << "./data/centroids_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		std::ifstream reader;
		reader.open(ss.str(), std::fstream::in);
		initialize_arrays(solver_type, size, is_boundary, face_normals, neighbor_ids, tr_areas, xbarys, ybarys);
		std::string line;
		for (int counter = 0; counter < size; counter++)
		{
			getline(reader, line);
			xbarys[counter] = std::stod(line);
			getline(reader, line);
			ybarys[counter] = std::stod(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/is_boundary_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int k = 0; k < size; k++)
		{
			getline(reader, line);
			is_boundary[k] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/neighbor_ids_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int k = 0; k < size; k++)
		{
			for (int i = 0; i < 3; i++)
			{
				getline(reader, line);
				int index = k*3+i;
				neighbor_ids[index] = stoi(line);
			}
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/face_normals_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str());
		
		for (int k = 0; k < size; k++)
		{
			for (int i = 0; i < 3; i++)
			{
				getline(reader, line);
				face_normals[2*(k*3+i)] = std::stod(line);
				getline(reader, line);
				face_normals[2*(k*3+i)+1] = std::stod(line);
			}
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/tr_areas_" << rank << ".txt";
		//std::cout << ss.str() << std::endl;
		reader.open(ss.str());
		
		for (int k = 0; k < size; k++)
		{
			getline(reader, line);
			tr_areas[k] = std::stod(line);
		}
	}

	void advection_2d::solver_small(double *&u, double const dt, double *&velocities, bool if_y, bool if_h, double t)
	{
		if (solver_type == 0)
		{
			double * interpolated_velocities;
			interpolated_velocities = new double [size * 3 * 2];
			solver_small_cpu(u, dt, velocities, interpolated_velocities, if_y, if_h, t);
			delete [] interpolated_velocities;
		}
		else
		{
			call_solver_small_cuda(u, dt, velocities, is_boundary, face_normals, neighbor_ids, tr_areas, xbarys, ybarys, size, if_y, if_h, t);
		}

	}

//mu_1 = 1.0, mu_2 = 1.0, epsilon_1 = 1.0, epsilon_2 = 2.25
	void advection_2d::solver_small_cpu(double *&u, double const &dt, double *&velocities, double *&interpolated_velocities, bool if_y, bool if_h, double t)
	{
		#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{	
			//if(t==0.0) {int id = omp_get_thread_num();

        		//printf("thread: %i\n",id);}
			for (int k = 0; k < 3; k++)
			{
				if (neighbor_ids[j*3+k] != -1)
				{
					if (if_h && if_y)
					{
						if (is_boundary[j]!=is_boundary[neighbor_ids[j*3+k]] && ybarys[j] < 0.9 && ybarys[j] > 0.1 && ybarys[neighbor_ids[j*3+k]] < 0.9 && ybarys[neighbor_ids[j*3+k]] > 0.1)
						{
							interpolated_velocities[2*(j*3+k)] = (is_boundary[j] ? 1.0 : -1.0)*cos(5.0*(xbarys[neighbor_ids[j*3+k]] - 3*t))+0.5*velocities[2*j]+velocities[2*neighbor_ids[j*3+k]]*0.5;
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]*0.5+velocities[2*neighbor_ids[j*3+k]+1]*0.5;
						}
						else
						{
							interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]]*0.5 + velocities[2*j]*0.5;
							interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1]*0.5 + velocities[2*j+1]*0.5;
						}
					}
					else
					{
						interpolated_velocities[2*(j*3+k)] = velocities[2*neighbor_ids[j*3+k]]*0.5 + velocities[2*j]*0.5;
						interpolated_velocities[2*(j*3+k)+1] = velocities[2*neighbor_ids[j*3+k]+1]*0.5 + velocities[2*j+1]*0.5;
					}
				}
				else
				{
					interpolated_velocities[2*(j*3+k)] = velocities[2*j]+0.0;
					interpolated_velocities[2*(j*3+k)+1] = velocities[2*j+1]+0.0;
				}
			}
		
		}
		#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{
			double temp = 0.0;
			for (int k = 0; k < 3; k++)
			{
				temp += interpolated_velocities[2*(3*j+k)] * face_normals[2*(j*3+k)] + interpolated_velocities[2*(3*j+k)+1] * face_normals[2*(j*3+k)+1]; 
			}
			if (!if_h) u[j] = u[j] - dt * (temp / tr_areas[j] + (if_y ? 0.5*pow(2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100)*u[j] : 0.5*pow(2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)*u[j]) )/(1.0+0.5*dt*( if_y ? pow(2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100) : pow(2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200) ) );
			else u[j] = u[j] - dt * (temp / tr_areas[j] + 0.5*pow(1.0/2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100)*u[j] + 0.5*pow(1.0/2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)*u[j])/(1.0+0.5*dt*( pow(1.0/2.25, 0.5)*sigma(ybarys[j], 0.01, 10, 100) + pow(1.0/2.25, 0.5)*sigma(xbarys[j], 0.01, 20, 200)));
		}
	}

	void advection_2d_mpi::solver_small(double *&u, double const &dt, double *&velocities, bool if_y, bool if_h, double t)
	{
		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0)
			{
				if ((solver_type==1) && (solver_types[i] == 1) && (rank != i))
				{
					call_set_send(u, velocities, send_data[i], ids_to_send[i], interfaces_size[i]);
				}
				else
				{
					for (int k = 0; k < interfaces_size[i]; k++)
					{
						send_data[i][3*k] = u[ids_to_send[i][k]];
						send_data[i][3*k+1] = velocities[2*ids_to_send[i][k]];
						send_data[i][3*k+2] = velocities[2*ids_to_send[i][k]+1];
					}
				}
			}
		}
		
		MPI_Request request[size_of_group];
		MPI_Request request_2[size_of_group];
		//U EXCHANGE HERE
		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0 && i!=rank) MPI_Isend(send_data[i], interfaces_size[i]*3, MPI_DOUBLE, i, rank, MPI_COMM_WORLD, &request[i]);
		}

		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0 && i!=rank) MPI_Irecv(receive_data[i], interfaces_size[i]*3, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &request_2[i]);
		}
		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0 && i!=rank)
			{
				MPI_Waitall(1, &request[i], MPI_STATUS_IGNORE);
				MPI_Waitall(1, &request_2[i], MPI_STATUS_IGNORE);
			}
		}

		//U EXCHANGE HERE END
		for (int i = 0; i < size_of_group; i++)
		{
			if (interfaces_size[i]!=0)
			{
				if ((solver_type==1) && (solver_types[i] == 1) && (rank != i))
				{
					call_set_receive(u, velocities, receive_data[i], received_ids[i], interfaces_size[i]);
				}
				else
				{
					for (int k = 0; k < interfaces_size[i]; k++)
					{
						u[received_ids[i][k]] = receive_data[i][3*k];
						velocities[2*received_ids[i][k]] = receive_data[i][3*k+1];
						velocities[2*received_ids[i][k]+1] = receive_data[i][3*k+2];
					}
				}
			}
		}
		advection_2d::solver_small(u, dt, velocities, if_y, if_h, t);
		MPI_Barrier(MPI_COMM_WORLD);
		
	}

	void advection_2d_mpi::calc_mpi_arrays()
	{
		//cout << "entering calcmpiarray" << endl;
		ids_to_send = new int * [size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			if ((solver_type==1) && (solver_types[i] == 1) && (i != rank)) call_cuda_malloc(ids_to_send[i], size*sizeof(int));
			else ids_to_send[i] = new int [size];
		}
		interfaces_size = new int [size_of_group];

		global_ids_reference = new int [size];

		std::stringstream ss;
		ss << "./data/global_ids_reference_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		std::ifstream reader;
		reader.open(ss.str(), std::fstream::in);
		std::string line;
		for (int j = 0; j < size; j++)
		{
			getline(reader, line);
			global_ids_reference[j] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/ids_send_size_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int j = 0; j < size_of_group; j++)
		{
			getline(reader, line);
			interfaces_size[j] = std::stoi(line);
		}
		reader.close();
		ss.str("");
		ss.clear();
		ss << "./data/ids_to_send_" << rank << ".txt";
		//std::cout << ss.str()<< std::endl;
		reader.open(ss.str(), std::fstream::in);
		for (int j = 0; j < size_of_group; j++)
			for (int i = 0; i < interfaces_size[j]; i++)
			{
				reader >> ids_to_send[j][i];
			}
		reader.close();

		received_ids = new int * [size_of_group];
		for (int i =0; i < size_of_group; i++)
		{
			if ((solver_type==1) && (solver_types[i] == 1) && (i != rank)) call_cuda_malloc(received_ids[i], interfaces_size[i]*sizeof(int));
			else received_ids[i] = new int [interfaces_size[i]];
		}
		MPI_Request request[size_of_group];
		for (int i = 0; i < size_of_group; i++)
		{
			if (i!=rank) MPI_Isend(ids_to_send[i], interfaces_size[i], MPI_INT, i, rank, MPI_COMM_WORLD, &request[i]);
		}
		for (int i = 0; i < size_of_group; i++)
		{
			if (i != rank)
			{
				int * buffer = new int [interfaces_size[i]];
				MPI_Recv(buffer, interfaces_size[i], MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < interfaces_size[i]; k++)
					{
						if(global_ids_reference[j] == buffer[k]) received_ids[i][k] = j;
					}	
				}
				for (int k = 0; k < interfaces_size[i]; k++)
				{
					for (int j = 0; j < size; j++)
					{
					
						if(global_ids_reference[j] == ids_to_send[i][k]) ids_to_send[i][k] = j;
					}	
				}
				delete [] buffer;
			}
		}
		send_data = new double * [size_of_group];
                receive_data = new double * [size_of_group];
                for (int i = 0; i < size_of_group; i++)
                {
                        if (interfaces_size[i]!=0)
                        {
                                if ((solver_type==1) && (solver_types[i] == 1) && (i != rank))
                                {
                                        call_alloc_transfer_array(send_data[i], receive_data[i], interfaces_size[i]*3);
                                }
                                else
                                {
                                        send_data[i] = new double [interfaces_size[i]*3];
                                        receive_data[i] = new double [interfaces_size[i]*3];    
                                }
                        }
                }
	}
}
