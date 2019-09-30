#ifndef _initialCost_KERNEL_H_
#define _initialCost_KERNEL_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define INITIALCOST_KERNEL_blockSize_x 32
#define INITIALCOST_KERNEL_blockSize_y 8

#define RADIUS 6
#define MAX_DISPARITY 51

#define CENSUS_WEIGHT 9


__forceinline__ __device__ unsigned long long int int2_as_ulonglong(int2 a)
{
	unsigned long long int res;
	asm("mov.b64 %0, {%1,%2};" : "=l"(res) : "r"(a.x), "r"(a.y));
	return res;
}

__global__ void initCostKernel_left(cudaPitchedPtr C, unsigned int w, unsigned int h, 
	cudaTextureObject_t greytex2Dleft, cudaTextureObject_t greytex2Dright, 
	cudaTextureObject_t ctex2Dleft, cudaTextureObject_t ctex2Dright)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidy >= h || tidx >= w) return;
	char* devPtr = (char *)C.ptr;
	size_t pitch = C.pitch;
	size_t slicePitch = pitch * h;
	unsigned char imLeft, imRight;
	int2 imLeft_ct, imRight_ct;
	unsigned long long int result;
	unsigned int diff;
	int i, j;
#pragma unroll
	for (unsigned int d = 0; d < MAX_DISPARITY; d++) {
		diff = 0;
#pragma unroll
		for (int i = -RADIUS; i < RADIUS; i++) {
#pragma unroll
			for (int j = -RADIUS; j < RADIUS; j++) {
				imLeft = tex2D<unsigned char>(greytex2Dleft, tidx + i, tidy + j);
				imRight = tex2D<unsigned char>(greytex2Dright, tidx - d + i, tidy + j);
				diff += (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
				imLeft_ct = tex2D<int2>(ctex2Dleft, tidx + i, tidy + j);
				imRight_ct = tex2D<int2>(ctex2Dright, tidx - d + i, tidy + j);
				result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
				/*
				if (tidx == 400 && tidy == 400 && i == 0 && j == 0 && d == 0) {
					printf("%llu %llu %llu\n", int2_as_ulonglong(imLeft_ct), int2_as_ulonglong(imRight_ct), result);
					printf("%d\n", __popcll(result));
				}*/
				diff += __popcll(result) * CENSUS_WEIGHT;
			}
		}
		/*
		if (tidx == 400 && tidy == 400 && d < 10) {
			printf("%lu %ld\n", d, diff);
		}
		*/
		char* slice = devPtr + d * slicePitch;
		unsigned int* row = (unsigned int*)(slice + tidy * pitch);
		row[tidx] = diff;
	}
}

__global__ void initCostKernel_right(cudaPitchedPtr C, unsigned int w, unsigned int h,
	cudaTextureObject_t greytex2Dleft, cudaTextureObject_t greytex2Dright,
	cudaTextureObject_t ctex2Dleft, cudaTextureObject_t ctex2Dright)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidy >= h || tidx >= w) return;
	char* devPtr = (char *)C.ptr;
	size_t pitch = C.pitch;
	size_t slicePitch = pitch * h;
	unsigned char imLeft, imRight;
	int2 imLeft_ct, imRight_ct;
	unsigned long long int result;
	unsigned int diff;
#pragma unroll
	for (unsigned int d = 0; d < MAX_DISPARITY; d++) {
		diff = 0;
#pragma unroll
		for (int i = -RADIUS; i < RADIUS; i++) {
#pragma unroll
			for (int j = -RADIUS; j < RADIUS; j++) {
				imLeft = tex2D<unsigned char>(greytex2Dleft, tidx + i, tidy + j);
				imRight = tex2D<unsigned char>(greytex2Dright, tidx + d + i, tidy + j);
				diff += (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
				imLeft_ct = tex2D<int2>(ctex2Dleft, tidx + i, tidy + j);
				imRight_ct = tex2D<int2>(ctex2Dright, tidx + d + i, tidy + j);
				result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
				/*
				if (tidx == 400 && tidy == 400 && i == 0 && j == 0 && d == 0) {
				printf("%llu %llu %llu\n", int2_as_ulonglong(imLeft_ct), int2_as_ulonglong(imRight_ct), result);
				printf("%d\n", __popcll(result));
				}*/
				diff += __popcll(result) * CENSUS_WEIGHT;
			}
		}
		/*
		if (tidx == 400 && tidy == 400 && d < 10) {
		printf("%lu %ld\n", d, diff);
		}
		*/
		char* slice = devPtr + d * slicePitch;
		unsigned int* row = (unsigned int*)(slice + tidy * pitch);
		row[tidx] = diff;
	}
}
#endif