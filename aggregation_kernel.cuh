#ifndef _AGGREGATION_KERNEL_H_
#define _AGGREGATION_KERNEL_H_

#define AGGREGATION_KERNEL_blockSize_x 32
#define AGGREGATION_KERNEL_blockSize_y 8

__global__ void aggregationKernel(unsigned char *d_D, cudaPitchedPtr L, unsigned int w, unsigned int h, const unsigned int Dmax, size_t pitches)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < w && tidy < h) {
		unsigned int minCost = 0xffffffff;
		unsigned int best_d = 0;
		char* LdevPtr = (char *)L.ptr;
		size_t Lpitch = L.pitch;
		size_t LslicePitch = Lpitch * h;
		char* Lslice;
		unsigned int* Lrow;
#pragma unroll
		for (unsigned int d = 0; d < Dmax; d++) {
			Lslice = LdevPtr + d * LslicePitch;
			Lrow = (unsigned int*)(Lslice + tidy * Lpitch);
			if (minCost > Lrow[tidx]) {
				minCost = Lrow[tidx];
				best_d = d;
			}
		}
		d_D[tidy * pitches + tidx] = (unsigned char)best_d;//((double)best_d / (double)Dmax*255.0);
	}
}


#endif