#include "colorbilateralfilter.hpp"


void initializePermutohedral(float * image, int H, int W, float sigmargb, int
DIM, Permutohedral & lattice_){
    float * features = new float[H * W * DIM];
	for( int j=0; j<H; j++ ){
		for( int i=0; i<W; i++ ){
		    int idx = j*W + i;
		    for( int z=0; z<DIM; z++ ){
			features[idx*DIM+z] = float(image[z*W*H + idx]) / sigmargb;
			}
		}
    }
    
    lattice_.init( features, DIM, H * W );
    delete [] features;
}


void colorbilateralfilter(float * image, int len_image, float * in, int len_in,
float * out, int len_out, int H, int W, float sigmargb, int DIM){
	Permutohedral lattice;
	initializePermutohedral(image, H, W, sigmargb, DIM, lattice);
    // number of classes
    int K = len_in/W/H;
    
    float * out_p = new float[W*H];
    float * in_p = new float[W*H];
    for(int k=0;k<K;k++){
        for(int i=0;i<W*H;i++)
            in_p[i] = in[i+k*W*H];
        lattice.compute(out_p, in_p, 1);
        for(int i=0;i<W*H;i++)
            out[i+k*W*H] = out_p[i];
    }
    delete [] out_p;
    delete [] in_p;
}

void colorbilateralfilter_batch(float * images, int len_images, float * ins, int
 len_ins, float * outs, int len_outs, int N, int K, int H, int W, float sigmargb, int DIM){

    const int maxNumThreads = omp_get_max_threads();
    //printf("Maximum number of threads for this machine: %i\n", maxNumThreads);
    omp_set_num_threads(std::min(maxNumThreads,N));

    #pragma omp parallel for
    for(int n=0;n<N;n++){
        colorbilateralfilter(images+n*3*H*W, 3*H*W, ins+n*K*H*W, K*H*W,
        outs+n*K*H*W, K*H*W, H, W, sigmargb, DIM);
    }
    //printf("parallel for\n");
}

