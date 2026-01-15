```sh
g++ -std=c++17 -DNDEBUG -O3 \
    -I /home/p2-admin/include \
    -L /home/p2-admin/lib \
    gen_sdsl_vectors.cpp \
    -lsdsl -ldivsufsort -ldivsufsort64 \
    -o gen_sdsl_vectors

./gen_sdsl_vectors <num_rows> <num_vectors> <output_dir> <prefix> <sparsity>
```
