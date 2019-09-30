/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/* Simple kernel computes a Stereo Disparity using CUDA SIMD SAD intrinsics. */

#ifndef _STEREODISPARITY_KERNEL_H_
#define _STEREODISPARITY_KERNEL_H_

#define STEREO_KERNEL_blockSize_x 32
#define STEREO_KERNEL_blockSize_y 8
#define MAX_DISPARITY 70

// RAD is the radius of the region of support for the search
// note: don't change this value because the cost implementation of this kernel code is dependent on RAD = blockSize_y = blockSize_x/4
#define RAD 8
#define CENSUS_WEIGHT 9
// STEPS is the number of loads we must perform to initialize the shared memory area
// (see convolution CUDA Sample for example)
#define STEPS 3

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// This function applies the video intrinsic operations to compute a
// sum of absolute differences.  The absolute differences are computed
// and the optional .add instruction is used to sum the lanes.
//
// For more information, see also the documents:
//  "Using_Inline_PTX_Assembly_In_CUDA.pdf"
// and also the PTX ISA documentation for the architecture in question, e.g.:
//  "ptx_isa_3.0K.pdf"
// included in the NVIDIA GPU Computing Toolkit
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ unsigned long long int int2_as_ulonglong(int2 a)
{
	unsigned long long int res;
	asm("mov.b64 %0, {%1,%2};" : "=l"(res) : "r"(a.x), "r"(a.y));
	return res;
}


////////////////////////////////////////////////////////////////////////////////
//! Simple stereo disparity kernel to test atomic instructions
//! Algorithm Explanation:
//! For stereo disparity this performs a basic block matching scheme.
//! The sum of abs. diffs between and area of the candidate pixel in the left images
//! is computed against different horizontal shifts of areas from the right.
//! The shift at which the difference is minimum is taken as how far that pixel
//! moved between left/right image pairs.   The recovered motion is the disparity map
//! More motion indicates more parallax indicates a closer object.
//! @param g_img1  image 1 in global memory, RGBA, 4 bytes/pixel
//! @param g_img2  image 2 in global memory
//! @param g_odata disparity map output in global memory,  unsigned int output/pixel
//! @param w image width in pixels
//! @param h image height in pixels
//! @param minDisparity leftmost search range
//! @param maxDisparity rightmost search range
////////////////////////////////////////////////////////////////////////////////
__global__ void
stereoDisparityKernel_left(unsigned char *g_odata,
	unsigned int w, unsigned int h, size_t pitches, 
	cudaTextureObject_t greytex2Dleft, cudaTextureObject_t greytex2Dright,
	cudaTextureObject_t ctex2Dleft, cudaTextureObject_t ctex2Dright)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	// access thread id
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int sidx = threadIdx.x + RAD;
	const unsigned int sidy = threadIdx.y + RAD;

	unsigned int imLeft;
	unsigned int imRight;
	int2 imLeft_ct, imRight_ct;
	unsigned int cost;
	unsigned long long result;
	unsigned int bestCost = 9999999;
	unsigned int bestDisparity = 0;
	__shared__ unsigned int diff[STEREO_KERNEL_blockSize_y + 2 * RAD][STEREO_KERNEL_blockSize_x + 2 * RAD];

	// store needed values for left image into registers (constant indexed local vars)
	unsigned char imLeftA_gray[STEPS];
	unsigned char imLeftB_gray[STEPS];
	int2 imLeftA_ct[STEPS];
	int2 imLeftB_ct[STEPS];

	for (int i = 0; i<STEPS; i++)
	{
		int offset = -RAD + i * RAD;
		imLeftA_gray[i] = tex2D<unsigned char>(greytex2Dleft, tidx - RAD, tidy + offset);
		imLeftB_gray[i] = tex2D<unsigned char>(greytex2Dleft, tidx - RAD + STEREO_KERNEL_blockSize_x, tidy + offset);
		imLeftA_ct[i] = tex2D<int2>(ctex2Dleft, tidx - RAD, tidy + offset);
		imLeftB_ct[i] = tex2D<int2>(ctex2Dleft, tidx - RAD + STEREO_KERNEL_blockSize_x, tidy + offset);
	}

	// for a fixed camera system this could be hardcoded and loop unrolled
	for (unsigned int d = 0; d <= MAX_DISPARITY; d++)
	{
		//LEFT
#pragma unroll
		for (int i = 0; i<STEPS; i++)
		{
			int offset = -RAD + i * RAD;
			//imLeft = tex2D( tex2Dleft, tidx-RAD, tidy+offset );
			imLeft = imLeftA_gray[i];
			imRight = tex2D<unsigned char>(greytex2Dright, tidx - RAD - d, tidy + offset);
			cost = (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
			imLeft_ct = imLeftA_ct[i];
			imRight_ct = tex2D<int2>(ctex2Dright, tidx - RAD - d, tidy + offset);
			result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
			diff[sidy + offset][sidx - RAD] = cost + __popcll(result) * CENSUS_WEIGHT;
		}

		//RIGHT
#pragma unroll
		for (int i = 0; i<STEPS; i++)
		{
			int offset = -RAD + i * RAD;
			if (threadIdx.x < 2 * RAD)
			{
				//imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
				imLeft = imLeftB_gray[i];
				imRight = tex2D<unsigned char>(greytex2Dright, tidx - RAD + STEREO_KERNEL_blockSize_x - d, tidy + offset);
				cost = (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
				imLeft_ct = imLeftB_ct[i];
				imRight_ct = tex2D<int2>(ctex2Dright, tidx - RAD + STEREO_KERNEL_blockSize_x - d, tidy + offset);
				result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
				diff[sidy + offset][sidx - RAD + STEREO_KERNEL_blockSize_x] = cost + __popcll(result) * CENSUS_WEIGHT;
			}
		}

		cg::sync(cta);

		// sum cost horizontally
#pragma unroll
		for (int j = 0; j<STEPS; j++)
		{
			int offset = -RAD + j * RAD;
			cost = 0;
#pragma unroll

			for (int i = -RAD; i <= RAD; i++)
			{
				cost += diff[sidy + offset][sidx + i];
			}

			cg::sync(cta);
			diff[sidy + offset][sidx] = cost;
			cg::sync(cta);
		}

		// sum cost vertically
		cost = 0;
#pragma unroll

		for (int i = -RAD; i <= RAD; i++)
		{
			cost += diff[sidy + i][sidx];
		}

		// see if it is better or not
		if (cost < bestCost)
		{
			bestCost = cost;
			bestDisparity = d;
		}

		cg::sync(cta);

	}

	if (tidy < h && tidx < w)
	{
		g_odata[tidy*pitches + tidx] = (unsigned char)bestDisparity;
	}
}

__global__ void
stereoDisparityKernel_right(unsigned char *g_odata,
	unsigned int w, unsigned int h, size_t pitches, 
	cudaTextureObject_t greytex2Dleft, cudaTextureObject_t greytex2Dright,
	cudaTextureObject_t ctex2Dleft, cudaTextureObject_t ctex2Dright)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	// access thread id
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int sidx = threadIdx.x + RAD;
	const unsigned int sidy = threadIdx.y + RAD;

	unsigned int imLeft;
	unsigned int imRight;
	int2 imLeft_ct, imRight_ct;
	unsigned int cost;
	unsigned long long result;
	unsigned int bestCost = 9999999;
	unsigned int bestDisparity = 0;
	__shared__ unsigned int diff[STEREO_KERNEL_blockSize_y + 2 * RAD][STEREO_KERNEL_blockSize_x + 2 * RAD];

	// store needed values for left image into registers (constant indexed local vars)
	unsigned int imLeftA_gray[STEPS];
	unsigned int imLeftB_gray[STEPS];
	int2 imLeftA_ct[STEPS];
	int2 imLeftB_ct[STEPS];

	for (int i = 0; i<STEPS; i++)
	{
		int offset = -RAD + i * RAD;
		imLeftA_gray[i] = tex2D<unsigned char>(greytex2Dleft, tidx - RAD, tidy + offset);
		imLeftB_gray[i] = tex2D<unsigned char>(greytex2Dleft, tidx - RAD + STEREO_KERNEL_blockSize_x, tidy + offset);
		imLeftA_ct[i] = tex2D<int2>(ctex2Dleft, tidx - RAD, tidy + offset);
		imLeftB_ct[i] = tex2D<int2>(ctex2Dleft, tidx - RAD + STEREO_KERNEL_blockSize_x, tidy + offset);
	}

	// for a fixed camera system this could be hardcoded and loop unrolled
	for (unsigned int d = 0; d <= MAX_DISPARITY; d++)
	{
		//LEFT
#pragma unroll
		for (int i = 0; i<STEPS; i++)
		{
			int offset = -RAD + i * RAD;
			//imLeft = tex2D( tex2Dleft, tidx-RAD, tidy+offset );
			imLeft = imLeftA_gray[i];
			imRight = tex2D<unsigned char>(greytex2Dright, tidx - RAD + d, tidy + offset);
			cost = (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
			imLeft_ct = imLeftA_ct[i];
			imRight_ct = tex2D<int2>(ctex2Dright, tidx - RAD + d, tidy + offset);
			result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
			diff[sidy + offset][sidx - RAD] = cost + __popcll(result) * CENSUS_WEIGHT;
		}

		//RIGHT
#pragma unroll
		for (int i = 0; i<STEPS; i++)
		{
			int offset = -RAD + i * RAD;
				if (threadIdx.x < 2 * RAD)
				{
					//imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
					imLeft = imLeftB_gray[i];
					imRight = tex2D<unsigned char>(greytex2Dright, tidx - RAD + STEREO_KERNEL_blockSize_x + d, tidy + offset);
					cost = (unsigned int)__usad((unsigned int)imLeft, (unsigned int)imRight, 0);
					imLeft_ct = imLeftB_ct[i];
					imRight_ct = tex2D<int2>(ctex2Dright, tidx - RAD + STEREO_KERNEL_blockSize_x + d, tidy + offset);
					result = int2_as_ulonglong(imLeft_ct) ^ int2_as_ulonglong(imRight_ct);
					diff[sidy + offset][sidx - RAD + STEREO_KERNEL_blockSize_x] = cost + __popcll(result) * CENSUS_WEIGHT;
				}
		}

		cg::sync(cta);

		// sum cost horizontally
#pragma unroll

		for (int j = 0; j<STEPS; j++)
		{
			int offset = -RAD + j * RAD;
			cost = 0;
#pragma unroll

			for (int i = -RAD; i <= RAD; i++)
			{
				cost += diff[sidy + offset][sidx + i];
			}

			cg::sync(cta);
			diff[sidy + offset][sidx] = cost;
			cg::sync(cta);
		}

		// sum cost vertically
		cost = 0;
#pragma unroll

		for (int i = -RAD; i <= RAD; i++)
		{
			cost += diff[sidy + i][sidx];
		}

		// see if it is better or not
		if (cost < bestCost)
		{
			bestCost = cost;
			bestDisparity = d;
		}

		cg::sync(cta);

	}

	if (tidy < h && tidx < w)
	{
		g_odata[tidy*pitches + tidx] = bestDisparity;
	}
}
#endif // #ifndef _STEREODISPARITY_KERNEL_H_
