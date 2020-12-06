# Parallel Programming implemented in CUDA

**Problem statement**: (Sparse matrix multiplication) 
In all the 4 assignments of my course (Information retrieval), the most expensive part of the algorithm was calculation of cosine similarity which was nothing but multiplication of 10,000-by-10,000 user rating matrix with 10,000-by-1 user vector.
I figured out that matrix multiplication on GPU which was implemented before was general(for dense matrices) and it could be again optimized for Sparse Matrices(as in my assignments).

**Solution**: 
The sparse matrix is first compressed and then the matrix parallelization code of GPU is performed. In  my code,
1. I compressed the matrix by using DIA,CSR and COO format, and
2. Implemented the parallelized matrix multiplication on GPU accordingly. I created  thread blocks of 16 thread each and wrote the code and kernel for each of the formats separately.