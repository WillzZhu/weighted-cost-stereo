#ifndef _CONSISTENCY_KERNEL_H_
#define _CONSISTENCY_KERNEL_H_

#define CONSISTENCY_KERNEL_blockSize_x 32
#define CONSISTENCY_KERNEL_blockSize_y 8

__global__ void consistencyKernel(unsigned char *dc, cudaTextureObject_t tex2Dleft, cudaTextureObject_t tex2Dright, 
	unsigned int w, unsigned int h, 
	size_t pitches, unsigned char threshold)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidy >= h || tidx >= w) return;
	unsigned char left_pixel = tex2D<unsigned char>(tex2Dleft, tidx, tidy);
	unsigned char right_pixel = tex2D<unsigned char>(tex2Dright, tidx-left_pixel, tidy);
	unsigned char diff = (unsigned char)__usad((unsigned int)left_pixel, (unsigned int)right_pixel, 0);
	dc[tidy * pitches + tidx] = diff >= threshold ? 0 : (unsigned char)(((double)left_pixel)*((double)(255.0/(double)MAX_DISPARITY)));
}
#endif