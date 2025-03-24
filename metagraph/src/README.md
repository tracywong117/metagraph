g++ -std=c++17 -O3 -DNDEBUG -I ~/include -L ~/lib row_diff_sequential.cpp -o row_diff_sequential -lsdsl -ldivsufsort -ldivsufsort64
g++ -std=c++17 -O3 -DNDEBUG -I ~/include -L ~/lib row_diff_kmean.cpp -o row_diff_kmean -lsdsl -ldivsufsort -ldivsufsort64
g++ -std=c++17 -O3 -DNDEBUG -I ~/include -L ~/lib row_diff_agglomerative.cpp -o row_diff_agglomerative -lsdsl -ldivsufsort -ldivsufsort64

g++ -std=c++17 -O3 -DNDEBUG -I ~/include -L ~/lib row_diff_agglomerative_omp.cpp -o row_diff_agglomerative_omp -lsdsl -ldivsufsort -ldivsufsort64 -fopenmp

g++ -std=c++17 -O3 -DNDEBUG -I ~/include -L ~/lib row_diff_sequential_ascending.cpp -o row_diff_sequential_ascending -lsdsl -ldivsufsort -ldivsufsort64

g++ -std=c++17 bit_vector_row_diff_sequential.cpp -o bit_vector_row_diff_sequential

g++ -std=c++17 bit_vector_row_diff_sequential_optimize.cpp -o bit_vector_row_diff_sequential_optimize -fopenmp