#include <math.h>
#include <vector>
#include <cstddef>
#include <stdlib.h>
#include <omp.h>
using namespace std;

#include "permutohedral.hpp"

void initializePermutohedral(float * image, int H, int W, float sigmargb, int
DIM, Permutohedral & lattice_);
void colorbilateralfilter(float * image, int len_image, float * in, int len_in,
float * out, int len_out, int H, int W, float sigmargb, int DIM);
void colorbilateralfilter_batch(float * images, int len_images, float * ins, int
len_ins, float * outs, int len_outs,int N, int K, int H, int W, float sigmargb,
int DIM);


