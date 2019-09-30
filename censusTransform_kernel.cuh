#ifndef _censusTransform_KERNEL_H_
#define _censusTransform_KERNEL_H_


#define CENSUSTRANSFORM_KERNEL_blockSize_x 32
#define CENSUSTRANSFORM_KERNEL_blockSize_y 8

// 9x7 census window
#define CENSUSTRANSFORM_WINDOW_X 4
#define CENSUSTRANSFORM_WINDOW_Y 3

__global__ void censusTransformKernel(unsigned long long *ct0, unsigned long long *ct1, 
									unsigned int w, unsigned int h, size_t ct_pitches_width,
									cudaTextureObject_t tex2Dleft,
									cudaTextureObject_t tex2Dright)
{
	// to do: shared memory
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned char imLeft = tex2D<unsigned char>(tex2Dleft, tidx, tidy);
	unsigned char imRight = tex2D<unsigned char>(tex2Dright, tidx, tidy);
	unsigned char cmp_pix;
	unsigned long long ctLeft = 0, ctRight = 0;
#pragma unroll
	for (int y = CENSUSTRANSFORM_WINDOW_Y; y >= -CENSUSTRANSFORM_WINDOW_Y; y--) {
#pragma unroll
		for (int x = CENSUSTRANSFORM_WINDOW_X; x >= -CENSUSTRANSFORM_WINDOW_X; x--) {
			cmp_pix = tex2D<unsigned char>(tex2Dleft, tidx + x, tidy + y);
			if (cmp_pix > imLeft) {
				ctLeft += 1;
			}
			cmp_pix = tex2D<unsigned char>(tex2Dright, tidx + x, tidy + y);
			if (cmp_pix > imRight) {
				ctRight += 1;
			}
			ctLeft = ctLeft << 1;
			ctRight = ctRight << 1;
		}
	}
	//if (tidx == 400 && tidy == 400) printf("%llu\n", ctLeft);
	if (tidy < h) {
		ct0[tidy*ct_pitches_width + tidx] = ctLeft;
		ct1[tidy*ct_pitches_width + tidx] = ctRight;
	}
}
#endif