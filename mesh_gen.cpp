#include <algorithm>
#include <sstream>
#include <memory>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <complex>
#include <cassert>
#include <utility>
#include <omp.h>
#include <ctime>
#include <cstdlib>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/Color.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/centroid.h>

#include <CGAL/Constrained_triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;

typedef CGAL::Exact_intersections_tag Tag;

typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds, Tag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;
typedef CGAL::Triangulation_2<K,Tds> Triangulation;
typedef CGAL::Triangle_2<K> Triangle;
typedef CGAL::Vector_2<K> Vector;
typedef CDT::Face Face;
typedef CDT::Face_handle Face_handle;
typedef CDT::Finite_faces_iterator Face_iterator;

int operator-(Face_iterator a, Face_iterator b) { int count = 0;  while(b!=a){b++; count++;} return count;}

Face_iterator &operator += (Face_iterator &a, const int i) {for (int num = 0; num < i; num++) a++; return a;}


Face_handle get_find(CDT& cdt, Face_handle face)
{
    int count = 0;
    for (auto itt(cdt.finite_faces_begin()); itt != cdt.finite_faces_end(); itt++)
    {
        Face_handle dd = itt;
        if (dd==face)
            return itt;
        count++;
    }
    //std::cout << "IT'S NOT FOUND" << std::endl;
    return nullptr;
}

void get_proportions(int length_x, int length_y, int size_of_group, int N_CUDA, double prp, double * proportions)
{
    for (int i = 0; i < size_of_group; i++) 
    {
        double x_length, y_length;
        if (N_CUDA==0 || size_of_group - N_CUDA == 0)
        {
            x_length = length_x;
            y_length = length_y;
        }
        else if (i < N_CUDA)
        {
            x_length = length_x*prp;
            y_length = length_y;
        }
        else
        {
            x_length = length_x - length_x*prp;
            y_length = length_y;
        }
        double x_len = x_length;
        double y_len = y_length;
        int x_times = 0;
        int y_times = 0;
        if (i<N_CUDA)
        {
            while ( (y_length/y_len)*(x_length/x_len) != N_CUDA)
            {
                if (x_len > y_len)
                {
                    x_len/=2.0;
                    x_times++;
                }
                else
                {
                    y_len/=2.0;
                    y_times++;
                }
            }
            proportions[4*i] = x_times!=0 ? x_len*(i%(x_times*2)) : 0.0;
            proportions[4*i+1] = x_times!=0 ? x_len+x_len*(i%(x_times*2)) : x_len;
            proportions[4*i+2] =x_times!=0 ?( y_len* ((int) (i/(x_times*2))) ): (y_len * i);
            proportions[4*i+3] = x_times!=0 ? ( y_len* ((int) (i/(x_times*2))) + y_len): (y_len + y_len * i);
        }
        else
        {
            while ( (y_length/y_len)*(x_length/x_len) != (size_of_group - N_CUDA) )
            {
                if (x_len > y_len)
                {
                    x_len/=2.0;
                    x_times++;
                }
                else
                {
                    y_len/=2.0;
                    y_times++;
                }
            }
            proportions[4*i] = (x_times!=0 ? x_len*(i%(x_times*2)) : 0.0) + length_x - x_length;
            proportions[4*i+1] = (x_times!=0 ? x_len+x_len*(i%(x_times*2)) : x_len) + length_x - x_length;
            proportions[4*i+2] = x_times!=0 ? ( y_len* ((int) ((i-N_CUDA)/(x_times*2))) ): (y_len * (i-N_CUDA) );
            proportions[4*i+3] = x_times!=0 ? ( y_len* ((int) ((i-N_CUDA)/(x_times*2))) +y_len ): (y_len * (i-N_CUDA) + y_len);
        }
    }
    for (int i = 0; i < size_of_group; i++) std::cout << "proprotions for " << i << ": " << proportions[4*i+0] << " "
	                 << proportions[4*i+1] << " " << proportions[4*i+2] << " " << proportions[4*i+3] << std::endl;
}

int get_distance(CDT& cdt, Face_handle face)
{
    int count = 0;
    for (auto itt(cdt.finite_faces_begin()); itt != cdt.finite_faces_end(); itt++)
    {
        Face_handle dd = itt;
        if (dd==face)
            return count;
        count++;
    }
    std::cout << "IT'S NOT FOUND" << std::endl;
    return count;
}

void mesh_mpi(CDT& cdt, double dr, double x_length, double y_length, int n_processes,
              double crit_border, double const * proportions,
              std::vector<std::vector<Face_handle>> & output)
{
    Vertex_handle va = cdt.insert(Point(0.0,0.0));
    Vertex_handle vb = cdt.insert(Point(x_length,0));
    Vertex_handle vc = cdt.insert(Point(x_length, y_length));
    Vertex_handle vd = cdt.insert(Point(0.0, y_length));
    cdt.insert_constraint(va, vb);
    cdt.insert_constraint(vb, vc);
    cdt.insert_constraint(vc, vd);
    cdt.insert_constraint(vd, va);

    int N_CIRCLE = 50;//30;//16;
    double radius = 0.1;

    int i = 0;
    double val = i*2*3.1416/N_CIRCLE;
    double next_val = (i+1)*2*3.1416/N_CIRCLE;
    Vertex_handle vc_first = cdt.insert(Point(radius*cos(val) + x_length/2.0, radius*sin(val) + y_length/2.0 ));
    Vertex_handle vn_first = cdt.insert(Point(radius*cos(next_val) + x_length/2.0, radius*sin(next_val) + y_length/2.0));
    cdt.insert_constraint(vc_first, vn_first);
    for (int i = 0; i <  N_CIRCLE-1; i++)
    {
        val = i*2*3.1416/N_CIRCLE;
        next_val = (i+1)*2*3.1416/N_CIRCLE;
        Vertex_handle vc = cdt.insert(Point(radius*cos(val) + x_length/2.0, radius*sin(val) + y_length/2.0 ));
        Vertex_handle vn = cdt.insert(Point(radius*cos(next_val) + x_length/2.0, radius*sin(next_val) + y_length/2.0));
        cdt.insert_constraint(vc, vn);
    }
    i = N_CIRCLE-1;
    val = i*2*3.1416/N_CIRCLE;
    Vertex_handle vc_last = cdt.insert(Point(radius*cos(val) + x_length/2.0, radius*sin(val) + y_length/2.0 ));
    cdt.insert_constraint(vc_last, vc_first);
    CGAL::refine_Delaunay_mesh_2(cdt, Criteria(dr, dr - dr*0.1));
    std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
    std::cout<<"TOTAL NUMBER OF FACES: " << cdt.number_of_faces()<<std::endl;

    double start = omp_get_wtime();
    int * proc_labels = new int [cdt.number_of_faces()];
    for (int i = 0; i < cdt.number_of_faces(); i++) proc_labels[i] = 0;		
    i = 0;
    Face_iterator itt(cdt.finite_faces_begin());
    for (itt = cdt.finite_faces_begin(); itt != cdt.finite_faces_end(); itt++)
    {
        auto val = CGAL::centroid(cdt.triangle(itt));
        for (int j = 0; j < n_processes; j++)
        {
            if (val.x()>=proportions[j*4] && val.x()<proportions[j*4+1] 
                && val.y()>=proportions[j*4+2] && val.y()<proportions[j*4+3]) proc_labels[i] = j+1;
        }
        i++;
    }
    std::cout << omp_get_wtime() - start << " is the time of setting up proportions" << std::endl;
    start = omp_get_wtime();
    int counter = 0;
    bool changes = true;
    while (changes && counter < 5)
    {
        changes = false;
        #pragma omp parallel private(itt) 
        {
            int k = -1;
            int num = omp_get_num_threads();
            int chunk_size = (cdt.number_of_faces()+num-1)/(num);
            bool first_iteration = true;
            #pragma omp for schedule(static, chunk_size)
            for(itt = cdt.finite_faces_begin(); itt < cdt.finite_faces_end(); itt++)
            {
                if (first_iteration)
                {
                    k = omp_get_thread_num()*chunk_size;
                    //THIS IS A NICE ASSERT TO MAKE SURE THAT EVERYTHING IS WORKING CORRECTLY
                    //#pragma critical
                    //{
                    //    printf("%d=%d %d/%d\n",k, get_distance(cdt, itt), 
                    //           omp_get_thread_num(), omp_get_num_threads());
                    //}
                    first_iteration = false;
                }
                else k++;
                if ( (abs(proportions[4*(proc_labels[k]-1)]-CGAL::centroid(cdt.triangle(itt)).x()) < crit_border) 
                     ||(abs(proportions[4*(proc_labels[k]-1)+1]-CGAL::centroid(cdt.triangle(itt)).x()) < crit_border)
                     || (abs(proportions[4*(proc_labels[k]-1)+2]-CGAL::centroid(cdt.triangle(itt)).y()) < crit_border)
                     || (abs(proportions[4*(proc_labels[k]-1)+3]-CGAL::centroid(cdt.triangle(itt)).y()) < crit_border)
                )
                {
                    if (proc_labels[k] > 0)
                    {
                        auto n1 = itt->neighbor(0);
                        auto n2 = itt->neighbor(1);
                        auto n3 = itt->neighbor(2);
                        if (cdt.is_infinite(n1) || cdt.is_infinite(n2) || cdt.is_infinite(n3)) continue;
                        int c1 = get_distance(cdt, n1);
                        int c2 = get_distance(cdt, n2);
                        int c3 = get_distance(cdt, n3);
                        if (proc_labels[c1] == proc_labels[k] && proc_labels[c1] == proc_labels[c2]) ;
                        else if (proc_labels[c2] == proc_labels[k] && proc_labels[c2] == proc_labels[c3]) ;
                        else if (proc_labels[c3] == proc_labels[k] && proc_labels[c1] == proc_labels[c3]) ;
                        else if (proc_labels[c1] == proc_labels[k] && proc_labels[c2] == proc_labels[c3]) 
                        {
                            proc_labels[k] = proc_labels[c2];
                            changes = true;
                        }
                        else if (proc_labels[c2] == proc_labels[k] && proc_labels[c1] == proc_labels[c3])
                        {
                            proc_labels[k] = proc_labels[c3];
                            changes = true;
                        }
                        else if (proc_labels[c3] == proc_labels[k] && proc_labels[c2] == proc_labels[c1])
                        {
                            proc_labels[k] = proc_labels[c1];
                            changes = true;
                        }
                    }
                    else
                        std::cout << "THIS IS BAD";
                }
            }
        }
        counter++;
        std::cout << "second counter " << counter << std::endl;
    }
    std::cout << omp_get_wtime() - start << " is the time of proc_labels completion" << std::endl;
    start = omp_get_wtime();

    std::vector<int> temp_ids_to_send[n_processes][n_processes];

    int k = 0;
    //#pragma omp parallel for
    for (itt = cdt.finite_faces_begin(); itt != cdt.finite_faces_end(); itt++)
    {
        if ( (abs(proportions[4*(proc_labels[k]-1)]-CGAL::centroid(cdt.triangle(itt)).x()) < crit_border) 
             ||(abs(proportions[4*(proc_labels[k]-1)+1]-CGAL::centroid(cdt.triangle(itt)).x()) < crit_border)
             || (abs(proportions[4*(proc_labels[k]-1)+2]-CGAL::centroid(cdt.triangle(itt)).y()) < crit_border)
             || (abs(proportions[4*(proc_labels[k]-1)+3]-CGAL::centroid(cdt.triangle(itt)).y()) < crit_border)
        )
        {
            for (int num = 0; num < 3; num++) if (!cdt.is_infinite(itt->neighbor(num)) )
            {
                int val = proc_labels[get_distance(cdt, itt->neighbor(num))];
                if (proc_labels[k] != val)
                    temp_ids_to_send[proc_labels[k]-1][val-1].push_back(k);
            }
        }
        k++;
    }
    k = 0;
    for (int rank = 0; rank < n_processes; rank++)
    {
        k = 0;
        std::stringstream ss1;
        ss1 << "./data/ids_to_send_" << rank << ".txt";
        std::ofstream ids_to_send;
        ids_to_send.open(ss1.str(), std::ios::out);

        std::stringstream ss3;
        ss3 << "./data/ids_send_size_" << rank << ".txt";
        std::ofstream ids_send_size;
        ids_send_size.open(ss3.str(), std::ios::out);
        std::vector<Face_handle> temp{};
        for (itt = cdt.finite_faces_begin(); itt != cdt.finite_faces_end(); itt++)
        {
            for (int num = 0; num < n_processes; num++)
            {
                auto global_tr = find(temp_ids_to_send[num][rank].begin(), temp_ids_to_send[num][rank].end(), k);
                if (global_tr!=temp_ids_to_send[num][rank].end()) temp.push_back(itt);
            }
            if (rank==proc_labels[k]-1) temp.push_back(itt);
                k++;
        }
        output.push_back(temp);
        for (int i = 0; i < n_processes; i++)
        {
            for (int j = 0; j < temp_ids_to_send[rank][i].size(); j++) ids_to_send << temp_ids_to_send[rank][i][j] << " ";
            ids_to_send << std::endl;
            ids_send_size << temp_ids_to_send[rank][i].size() << std::endl;
        }

        ids_to_send.close();
        ids_send_size.close();
    }
    std::cout << omp_get_wtime() - start << " is time for filling ids" << std::endl;
}


int main(int argc, char ** argv)
{
    int size_of_group;
    double length_x, length_y;
    int N_CUDA;
    double prp;
    double dr;
    if(argc !=2) {std::cout << "need filename" << std::endl; return 2;}

    std::string filename{argv[1]};
    std::ifstream arguments;
    arguments.open(filename);
    std::vector<std::string> data;
    std::string line;
    int i = 0;
    while(getline(arguments, line))
    {
        data.push_back(line);
        i++;
    }
    if (data.size() != 7) {std::cout << "wrong data" << std::endl; return 2;}
    length_x = std::stod(data[0]);//2.0
    length_y = std::stod(data[1]);//0.5
    N_CUDA = std::stoi(data[2]);
    prp = std::stod(data[3]);
    dr = std::stod(data[4]);
    size_of_group = std::stoi(data[5]);
    double crit_border = std::stod(data[6]);
    double * proportions = new double[4*size_of_group];
    double start = omp_get_wtime();
    get_proportions(length_x, length_y, size_of_group, N_CUDA, prp, proportions);
    std::cout << omp_get_wtime() - start << " is getting proportions time" << std::endl;	

    CDT cdt;
    std::vector<std::vector<Face_handle>> visual_core;
    std::vector<Face_handle> temp{};
    //for (int i = 0; i < size_of_group; i++) visual_core.push_back(temp);
    start = omp_get_wtime();
    mesh_mpi(cdt, dr, length_x, length_y, size_of_group, crit_border, proportions, visual_core);
    std::cout << omp_get_wtime() - start << " is mesh_mpi time" << std::endl;

    start = omp_get_wtime();
    std::ofstream verts;
    verts.open("./data/vertices.txt", std::ios::out);
    for (auto it(cdt.points_begin()); it != cdt.points_end(); it++)
    {
        verts << (it)->x() << " " << (it)->y() << " 0.0" << std::endl; 
    }
    verts.close();
    Face_iterator it = cdt.finite_faces_begin();
  	
    #pragma omp parallel
    {
        std::stringstream ss;
        ss << "./data/faces_" << omp_get_thread_num() << ".txt";
        std::ofstream faces;
        faces.open(ss.str(), std::ios::out);
        #pragma omp for		
        for (it=cdt.finite_faces_begin(); it < cdt.finite_faces_end(); it++)
        {
            for (int i = 0; i < 3; i++)
            {
                auto vert = std::find(cdt.points_begin(), cdt.points_end(), cdt.triangle(it)[i]);
                if (vert != cdt.points_end()) faces << std::distance(cdt.points_begin(), vert) << " ";
                else std::cout << "SOMETHING HAPPENED" << std::endl;
            }
            faces << std::endl;
        }
        faces.close();
    }
    std::cout << omp_get_wtime() - start << " is time for writing vertices.txt and faces.txt" << std::endl;
    auto itt = visual_core[0].begin();
    start = omp_get_wtime();
    #pragma omp parallel for private(itt)
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/is_boundary_" << i << ".txt";
        std::ofstream is_boundary;
        is_boundary.open(ss.str(), std::ofstream::out);
        for (itt = visual_core[i].begin(); itt != visual_core[i].end(); itt++)
        {
            if (CGAL::centroid(cdt.triangle(*itt)).x() < 0.2) is_boundary << 1 << std::endl;
            else is_boundary << 0 << std::endl;
        }
        is_boundary.close();
    }

    #pragma omp parallel for private(itt)	
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/neighbor_ids_" << i << ".txt";
        std::ofstream neighbor_ids;
        neighbor_ids.open(ss.str(), std::ofstream::out);
        for (itt = visual_core[i].begin(); itt != visual_core[i].end(); itt++)
        {
            for (int j = 0; j < 3; j++)
            {
                auto opposite_tr = (*itt)->neighbor(j);
                auto neighbor = find(visual_core[i].begin(), visual_core[i].end(), opposite_tr);
                if ((opposite_tr == nullptr) || (neighbor == visual_core[i].end()))
                    neighbor_ids << -1 << std::endl;
                else
                    neighbor_ids << distance(visual_core[i].begin(), neighbor) << std::endl;
            }
        }
        neighbor_ids.close();
    }

    #pragma omp parallel for private(itt)
    for (int j = 0; j < size_of_group; j++)
    {
        std::stringstream ss;
        ss << "./data/face_normals_" << j << ".txt";
        std::ofstream face_normals;
        face_normals.open(ss.str(), std::ofstream::out);

        for (itt = visual_core[j].begin(); itt != visual_core[j].end(); itt++)
        {
            Point barry = CGAL::centroid(cdt.triangle(*itt));
            for (int i = 0; i < 3; i++)
            {
                Point corner1 = cdt.triangle(*itt)[(i+1)%3];
                Point corner2 = cdt.triangle(*itt)[(i+2)%3];
                Point center = Point((corner1.x() + corner2.x())/2.0, (corner1.y() + corner2.y())/2.0);
                double length = pow(pow(corner1.x()-corner2.x(), 2) + pow(corner1.y() - corner2.y(), 2), 0.5);
                double x1, y1;
                double x2, y2;
                if (fabs(corner1.y()-corner2.y())<=1e-6)
                {
                    x1 = x2 = center.x();
                    y1 = center.y() + length;
                    y2 = center.y() - length;
                }
                else if (fabs(corner1.x()-corner2.x())<=1e-6)
                {
                    y1 = y2 = center.y();
                    x1 = center.x() + length;
                    x2 = center.x() - length;
                }
                else
                {
                    double n_x = (corner1.y()-corner2.y())/(corner2.x()-corner1.x());
                    y1 = center.y() + pow(length*length/(1.0 + n_x*n_x),0.5);
                    y2 = center.y() - pow(length*length/(1.0 + n_x*n_x),0.5);
                    x1 = center.x() + n_x * (y1 - center.y());
                    x2 = center.x() + n_x * (y2 - center.y());
                }
                Vector dist1 = Vector(x1 - barry.x(), y1 - barry.y());
                Vector dist2 = Vector(x2 - barry.x(), y2 - barry.y()); 
                Point dest = (dist2.squared_length() > dist1.squared_length()) ? Point(x2, y2) : Point(x1, y1);
                face_normals << dest.x()-center.x() << std::endl;
                face_normals << dest.y()-center.y() << std::endl;
            }
        }
        face_normals.close();
    }
	
    #pragma omp parallel for private(itt)
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/tr_areas_" << i << ".txt";
        std::ofstream tr_areas;
        tr_areas.open(ss.str(), std::ofstream::out);
        for (itt = visual_core[i].begin(); itt != visual_core[i].end(); itt++)
            tr_areas << cdt.triangle(*itt).area() << std::endl;
        tr_areas.close();
    }

    #pragma omp parallel for
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/global_ids_reference_" << i << ".txt";
        std::ofstream global_ids;
        global_ids.open(ss.str(), std::ofstream::out);
        for (int j = 0; j < visual_core[i].size(); j++)
        {
            Face_handle global_tr = get_find(cdt, visual_core[i][j]);
            global_ids << get_distance(cdt, global_tr) << std::endl;	
        }
        global_ids.close();
    }
	
    #pragma omp parallel for
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/sizes_" << i << ".txt";
        std::ofstream sizes;
        sizes.open(ss.str(), std::ofstream::out);
        sizes << cdt.number_of_faces() << std::endl << visual_core[i].size() << std::endl;
        sizes.close();
    }
	
    #pragma omp parallel for
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/colours_" << i << ".txt";
        std::ofstream colours;
        colours.open(ss.str(), std::ofstream::out);
        for (int j = 0; j < visual_core[i].size(); j++)
        {
            float p_new = ((float) i) / ((float) size_of_group);
            if ((p_new) <= 0.25) colours << 0.0 << " " << ((p_new)/0.25)  << " " << 1.0 << " " << 1.0 << " "; 
            else if ((p_new) <= 0.5) colours << 0.0 << " " << 1.0 << " " << ((0.5-(p_new))/0.25) << " " << 1.0 << " ";
            else if ((p_new) <= 0.75) colours << (((p_new)-0.5)/0.25) << " " << 1.0 << " " << 0.0 << " " << 1.0 << " ";
            else colours << 1.0 << " " << ((1.0-p_new)/0.25) << " " << 0.0 << " " << 1.0 << " ";	
            colours << std::endl;	
        }
        colours.close();
    }

    #pragma omp parallel for		
    for (int i = 0; i < size_of_group; i++)
    {
        std::stringstream ss;
        ss << "./data/centroids_" << i << ".txt";
        std::ofstream centroids;
        centroids.open(ss.str(), std::ofstream::out);
        for (int j = 0; j < visual_core[i].size(); j++)
        {
            auto val = CGAL::centroid(cdt.triangle(visual_core[i][j]));
            centroids << val.x() << std::endl << val.y() << std::endl;
        }
        centroids.close();
    }
    std::cout << omp_get_wtime() - start << " is time of drawing colours" << std::endl;	
    return 0;
}

