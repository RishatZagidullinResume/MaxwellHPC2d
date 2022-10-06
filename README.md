# Maxwell equation solver in 2D using finite volume approach. HPC implementation.

## Code description

Based on [this publication](https://link.springer.com/article/10.1007/s10598-020-09496-6).

The code solves a scattering problem setting in 2D. The space domain is a triangulated mesh. We use finite volume approach to find the numerical solution.

* Parameters:
`dt` - time step, 
`TIME` - total number of iterations, 
`TOTAL_FRAMES` - number of iterations that will be dumped to a file,
`epsilon_` - electric permittivity values of the circle and the space, 
`mu_` - magnetic permeability values of the circle and the space, 
`N_CUDA` - number of processes that will solve the equation on GPU, 
`N_THREADS_PER_PROC` - number of threads per process, 
`dif_coefs` - diffusion coefficients array for different particle sizes, 
`initial_layer` - array with the initial condition (zero by default).

* The progam solves Maxwell equations in a hybrid parallel manner. Meaning that the computational domain can be divided and distributed to different processes. A processes can use GPU to solve the equation on the given subdomain. The program relies on **CUDA** and **MPI** technologies for parallelism.

* Another library that the program relies on is **CGAL** for mesh triangulation.

* There is a separate domain generation script (__mesh_gen.cpp__). To launch it use the __input_mesh__ input file (pass it as a command line argument). The program works in parallel using **OpenMP**. Essentially, one can deside how much space is given to GPU-backed and CPU-backed processes. For example see image below:

<p align="center" width="100%">
    <img width="80%" src="/mpi_domain.png?raw=true"> 
</p>

GPU in general solves Maxwell equation faster, so it's reasonable to give it a bigger chunk of the computational domain. If you have free CPU processors, they can also do a bit of work. However keep in mind that data exchange between processes is an overhead. It will be cancelled out only for sufficiently big computational domains. In the example above we see that half of the computational domain is given to a single GPU-backed process while the other half is equally distributed among the four CPU-backed processes.

* The numerical solver also takes a config file as a sole command line argument. See __input_solver__ for reference. Use `mpirun` to launch the executable.

* The __Makefile__ outputs the mesh generation executable (__mesh_gen.exe__) as well as the numerical solver executable (__2D_MPI_CUDA.exe__).

## Visualization and demo

* For visualization purposes see __2d_py.py__ script. It uses **PyQt5** for a simple GUI interactable and **pyopengl** for visualization. The script accepts 3 command line arguments. For example, `python3 2d_py.py 100 8 4` meaning: 100 time iterations are going to be visualized, mesh was generated using 8 OpenMP threads and the numerical solution was performed by 4 MPI processes.

* Below is an example of the numerical solution.

<p align="center" width="100%">
    <img width="80%" src="/image.gif?raw=true"> 
</p>


