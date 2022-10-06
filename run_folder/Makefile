GCC = g++
MPICC = mpicxx
NVCC = nvcc

CUDA_INSTALL_PATH = /usr/local/cuda
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
CUDALDFLAGS=-L${CUDA_INSTALL_PATH}/lib64
CUDA_LIB = -lcudart


MPI_INSTALL_PATH = /usr/lib/x86_64-linux-gnu/openmpi
MPICFLAGS=-I${MPI_INSTALL_PATH}/include
MPILDFLAGS=-L${MPI_INSTALL_PATH}/lib
MPI_LIB = -lmpi -lmpi_cxx

LIB  = -lgomp -lm -lgmp #-lCGAL

GCC_COPT = -fopenmp
NVCC_COPT = -Xcompiler -fopenmp

VPATH = ../advection:../

MESH_SOURCE = mesh_gen.cpp
MESH_OBJECT = $(MESH_SOURCE:.cpp=_cpp.o)

SOURCES = transport.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=_cpp.o)

CUDA_SOURCES = transport.cu helper.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=_cuda.o)

all: mesh_gen.exe 2D_MPI_CUDA.exe

mesh_gen.exe: $(MESH_OBJECT)
	$(GCC) $^ -o $@ $(LIB)

2D_MPI_CUDA.exe: $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $^ -o $@ $(LIB) $(NVCC_COPT) $(MPILDFLAGS) $(MPI_LIB)

%_cpp.o: %.cpp
	$(MPICC) -c $< -o $@ $(GCC_COPT) $(CUDACFLAGS) $(MPICFLAGS) 

%_cuda.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCC_COPT) $(MPICFLAGS)

clean:
	rm -rf *.o
	rm -rf *.exe
