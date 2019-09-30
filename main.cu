// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, kernels
#include <cuda_runtime.h>
#include "greyscale_kernel.cuh"
#include "censusTransform_kernel.cuh"
#include "stereoDisparity_kernel.cuh"
#include "consistency_kernel.cuh"


// includes helper
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

// png loading
#include "lodepng.h"
#define checkPNGError(png_load_error) { 	if (png_load_error) { fprintf(stderr, "error %u: %s\n", png_load_error, lodepng_error_text(png_load_error)); exit(EXIT_FAILURE);} }

//#define REALSENSE_INPUT

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void displayPNGInfo(const LodePNGInfo& info);

/* stereo workflow */
void runGreyscale(int argc, char **argv, unsigned char **img0, unsigned char **img1, unsigned int *w, unsigned int *h, size_t *pitches);
void runCensusTransform(unsigned char *img0, unsigned char *img1,
	unsigned long long **ct0, unsigned long long **ct1,
	unsigned int w, unsigned int h, size_t gc_pitches, size_t *ct_pitches);
void runStereoDisparity(unsigned char view_type, unsigned char *h_d, unsigned char **disp, unsigned char *img0, unsigned char *img1, unsigned long long *ct0, unsigned long long *ct1, unsigned int w, unsigned int h, size_t gc_pitches, size_t ct_pitches);
void runConsistencyCheck(unsigned char *h_dc, unsigned char *d_left, unsigned char *d_right, size_t pitches, unsigned int w, unsigned int h, unsigned char threshold);

int main(int argc, char **argv) {
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;
	int dev = 0;

	// This will pick the best possible CUDA capable device
	dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	/* image parameters */
	unsigned int w, h;
	size_t gc_pitches, ct_pitches;

	/* First stage: Greyscale left & right image */
	unsigned char **img0, **img1; // device ptr to greyscale output
	img0 = (unsigned char **)malloc(sizeof(unsigned char *));
	img1 = (unsigned char **)malloc(sizeof(unsigned char *));
	runGreyscale(argc, argv, img0, img1, &w, &h, &gc_pitches);


	/* Second stage: 64-it Census Transform */
	unsigned long long **ct0, **ct1; // device ptr to census transform output
	ct0 = (unsigned long long **)malloc(sizeof(unsigned long long *));
	ct1 = (unsigned long long **)malloc(sizeof(unsigned long long *));
	runCensusTransform(*img0, *img1, ct0, ct1, w, h, gc_pitches, &ct_pitches);

	/* Third stage: Construct [W x H x Dmax] initial cost table */
	/* base image: left */
	/* Fourth stage: Cost Aggregation and calculate D[p] */
	unsigned char *h_D_left = (unsigned char *)malloc(sizeof(unsigned char) * w * h);
	unsigned char *h_D_right = (unsigned char *)malloc(sizeof(unsigned char) * w * h);
	unsigned char *d_left, *d_right;

	runStereoDisparity(0, h_D_left, &d_left, *img0, *img1, *ct0, *ct1, w, h, gc_pitches, ct_pitches);
	runStereoDisparity(1, h_D_right, &d_right, *img1, *img0, *ct1, *ct0, w, h, gc_pitches, ct_pitches);
	checkCudaErrors(cudaFree(*img0));
	checkCudaErrors(cudaFree(*img1));
	checkCudaErrors(cudaFree(*ct0));
	checkCudaErrors(cudaFree(*ct1));


	/* write disparity into png file */
#ifdef REALSENSE_INPUT
	unsigned png_load_error = lodepng_encode_file("./cuda-out/cuda-disp-left.png", h_D_left, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
	png_load_error = lodepng_encode_file("./cuda-out/cuda-disp-right.png", h_D_right, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
#else
	if (argc < 5) {
		fprintf(stderr, "Input Error: provide two png files for left/right output\n");
		exit(EXIT_FAILURE);
	}
	unsigned png_load_error = lodepng_encode_file(argv[3], h_D_left, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
	png_load_error = lodepng_encode_file(argv[4], h_D_right, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
#endif
	free(h_D_left);
	free(h_D_right);

	/* Consistency Checking */
	unsigned char *h_dc = (unsigned char *)malloc(sizeof(unsigned char) * w * h);
	runConsistencyCheck(h_dc, d_left, d_right, gc_pitches, w, h, 1);

#ifdef REALSENSE_INPUT
	png_load_error = lodepng_encode_file("./cuda-out/cuda-disp-consistency.png", h_dc, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
#else
	if (argc < 6) {
		fprintf(stderr, "Input Error: provide the png file for consistency map\n");
		exit(EXIT_FAILURE);
	}
	png_load_error = lodepng_encode_file(argv[5], h_dc, w, h, LCT_GREY, 8);
	checkPNGError(png_load_error);
#endif


	/* memory clean-up */
	checkCudaErrors(cudaFree(d_left));
	checkCudaErrors(cudaFree(d_right));
	free(img0);
	free(img1);
	free(h_dc);
	free(ct0);
	free(ct1);

	exit(EXIT_SUCCESS);
}

void displayPNGInfo(const LodePNGInfo& info) {
	const LodePNGColorMode& color = info.color;
	printf("Color type: %d (0: greyscale, 2: RGB, 3: palette, 4: GA, 6: RGBA)\n", color.colortype);
	printf("Bit depth: %d\n", color.bitdepth);
	printf("Bits per pixel: %d\n", lodepng_get_bpp(&color));
}

typedef enum TextureDataType {
	TDT_U8 = 1,
	TDT_U32 = 4,
	TDT_U64 = 8
} TextureDataType_t;

void init2DTextureObj(cudaTextureObject_t *tex, TextureDataType_t dataType, size_t pitches, void *devPtr, unsigned int w, unsigned int h) {
	cudaChannelFormatDesc ca_desc;
	switch (dataType) {
	case TDT_U8:
		ca_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		break;
	case TDT_U32:
		ca_desc = cudaCreateChannelDesc<unsigned int>();
		break;
	case TDT_U64:
		// store 64-bit as <int2, int2> type
		ca_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
		break;
	default:
		ca_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		break;
	}
	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypePitch2D;
	texRes.res.pitch2D.devPtr = devPtr;
	texRes.res.pitch2D.desc = ca_desc;
	texRes.res.pitch2D.width = pitches / (unsigned int)dataType;
	texRes.res.pitch2D.height = h;
	texRes.res.pitch2D.pitchInBytes = pitches;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(tex, &texRes, &texDescr, NULL));
}

void runGreyscale(int argc, char **argv, unsigned char **img0, unsigned char **img1, unsigned int *w, unsigned int *h, size_t *pitches) {
	// load png images into host memory
	unsigned char *h_img0 = NULL;
	unsigned char *h_img1 = NULL;
	unsigned char *png = NULL;
	size_t pngsize;
	unsigned png_load_error;
	LodePNGState png_state;
	
#ifdef REALSENSE_INPUT
	char *fname0 = "./data/rs-rgb-left.png";
	char *fname1 = "./data/rs-rgb-right.png";
#else
	char *fname0, *fname1;
	if (argc >= 3) {
		fname0 = argv[1];
		fname1 = argv[2];
	}
	else {
		fprintf(stderr, "Input Error: provide two png files and two output greyscale files\n");
		exit(EXIT_FAILURE);
	}
#endif
	lodepng_state_init(&png_state);
	png_load_error = lodepng_load_file(&png, &pngsize, fname0);
	checkPNGError(png_load_error);
	png_load_error = lodepng_decode(&h_img0, w, h, &png_state, png, pngsize);
	checkPNGError(png_load_error);
	printf("Loaded <%s> as image 0\n", fname0);
	png_load_error = lodepng_decode32_file(&h_img1, w, h, fname1);
	checkPNGError(png_load_error);
	printf("Loaded <%s> as image 1\n", fname1);
	displayPNGInfo(png_state.info_png);
	printf("Image size: [%d x %d]\n", *w, *h);
	free(png);

	/* only allows 4-byte alignment */
	LodePNGColorType ct = png_state.info_png.color.colortype;
	if (ct != LCT_RGB && ct != LCT_RGBA && png_state.info_png.color.bitdepth != 8) {
		printf("not supported image type\n");
		exit(EXIT_SUCCESS);
	}

	unsigned int numData = (*w) * (*h);
	unsigned int memSize = numData * sizeof(unsigned char);

	//allocate mem for the greyscale result on host side
	unsigned char *h_odata0 = (unsigned char *)malloc(memSize);
	unsigned char *h_odata1 = (unsigned char *)malloc(memSize);
	//cpu_greyscale(h_img0, h_odata, w, h);

	/* device memory */
	unsigned int *d_img0, *d_img1;
	unsigned char *d_odata0, *d_odata1; // greyscale output
	size_t pitches_in;
	checkCudaErrors(cudaMallocPitch((void **)&d_img0, &pitches_in, sizeof(unsigned int)*(*w), (*h)));
	checkCudaErrors(cudaMemcpy2D(d_img0, pitches_in, h_img0, sizeof(unsigned int)*(*w), sizeof(int)*(*w), (*h), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMallocPitch((void **)&d_img1, &pitches_in, sizeof(unsigned int)*(*w), (*h)));
	checkCudaErrors(cudaMemcpy2D(d_img1, pitches_in, h_img1, sizeof(unsigned int)*(*w), sizeof(int)*(*w), (*h), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMallocPitch((void **)&d_odata0, pitches, sizeof(unsigned char)*(*w), (*h)));
	checkCudaErrors(cudaMallocPitch((void **)&d_odata1, pitches, sizeof(unsigned char)*(*w), (*h)));


	dim3 numThreads = dim3(GREYSCALE_KERNEL_blockSize_x, GREYSCALE_KERNEL_blockSize_y, 1);
	dim3 numBlocks = dim3(iDivUp(*pitches, numThreads.x), iDivUp(*h, numThreads.y));

	/* texture */
	cudaTextureObject_t tex2Dleft, tex2Dright;
	init2DTextureObj(&tex2Dleft, TDT_U32, pitches_in, d_img0, *w, *h);
	init2DTextureObj(&tex2Dright, TDT_U32, pitches_in, d_img1, *w, *h);


	greyscaleKernel << <numBlocks, numThreads >> >(d_img0, d_img1, d_odata0, d_odata1, *pitches, tex2Dleft, tex2Dright);
	checkCudaErrors(cudaDeviceSynchronize());

	// greyscale output
	checkCudaErrors(cudaMemcpy2D(h_odata0, sizeof(unsigned char)*(*w), d_odata0, *pitches, sizeof(unsigned char)*(*w), (*h), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(h_odata1, sizeof(unsigned char)*(*w), d_odata1, *pitches, sizeof(unsigned char)*(*w), (*h), cudaMemcpyDeviceToHost));
	//png_load_error = lodepng_encode_file(greyscale_out0, h_odata0, *w, *h, LCT_GREY, 8);
	//checkPNGError(png_load_error);
	//png_load_error = lodepng_encode_file(greyscale_out1, h_odata1, *w, *h, LCT_GREY, 8);
	//checkPNGError(png_load_error);


	// device memory clean up
	//checkCudaErrors(cudaFree(d_odata0));
	//checkCudaErrors(cudaFree(d_odata1));
	*img0 = d_odata0;
	*img1 = d_odata1;
	// greyscale output passed to next step
	checkCudaErrors(cudaFree(d_img0));
	checkCudaErrors(cudaFree(d_img1));

	// host memory clean up
	lodepng_state_cleanup(&png_state);
	free(h_img0);
	free(h_img1);
	free(h_odata0);
	free(h_odata1);
}


void runCensusTransform(unsigned char *img0, unsigned char *img1, unsigned long long **ct0, unsigned long long **ct1, unsigned int w, unsigned int h, size_t gc_pitches, size_t *ct_pitches) {
	checkCudaErrors(cudaMallocPitch((void **)ct0, ct_pitches, sizeof(unsigned long long)*w, h));
	checkCudaErrors(cudaMallocPitch((void **)ct1, ct_pitches, sizeof(unsigned long long)*w, h));
	dim3 numThreads = dim3(CENSUSTRANSFORM_KERNEL_blockSize_x, CENSUSTRANSFORM_KERNEL_blockSize_y, 1);
	dim3 numBlocks = dim3(iDivUp((*ct_pitches) / 8, numThreads.x), iDivUp(h, numThreads.y)); // use output data size

																							 /* texture */
	cudaTextureObject_t tex2Dleft, tex2Dright;
	init2DTextureObj(&tex2Dleft, TDT_U8, gc_pitches, img0, w, h);
	init2DTextureObj(&tex2Dright, TDT_U8, gc_pitches, img1, w, h);

	/* run kernel */
	censusTransformKernel << <numBlocks, numThreads >> > (*ct0, *ct1, w, h, (*ct_pitches) / 8, tex2Dleft, tex2Dright);
	checkCudaErrors(cudaDeviceSynchronize());
}

void runStereoDisparity(unsigned char view_type, unsigned char *h_d, unsigned char **disp, unsigned char *img0, unsigned char *img1, unsigned long long *ct0, unsigned long long *ct1, unsigned int w, unsigned int h, size_t gc_pitches, size_t ct_pitches) {
	/* device memory */
	size_t d_pitches;
	checkCudaErrors(cudaMallocPitch((void **)disp, &d_pitches, sizeof(unsigned char)*w, h));

	dim3 numThreads = dim3(STEREO_KERNEL_blockSize_x, STEREO_KERNEL_blockSize_y, 1);
	dim3 numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));

	/* texture */
	cudaTextureObject_t greytex2Dleft, greytex2Dright;
	init2DTextureObj(&greytex2Dleft, TDT_U8, gc_pitches, img0, w, h);
	init2DTextureObj(&greytex2Dright, TDT_U8, gc_pitches, img1, w, h);

	cudaTextureObject_t ctex2Dleft, ctex2Dright;
	init2DTextureObj(&ctex2Dleft, TDT_U64, ct_pitches, ct0, w, h);
	init2DTextureObj(&ctex2Dright, TDT_U64, ct_pitches, ct1, w, h);

	if (view_type == 0) {
		stereoDisparityKernel_left << <numBlocks, numThreads >> > (*disp, w, h, d_pitches, 
			greytex2Dleft, greytex2Dright,
			ctex2Dleft, ctex2Dright);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else {
		stereoDisparityKernel_right << <numBlocks, numThreads >> > (*disp, w, h, d_pitches, 
			greytex2Dleft, greytex2Dright,
			ctex2Dleft, ctex2Dright);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaMemcpy2D(h_d, sizeof(unsigned char)*w, *disp, d_pitches, sizeof(unsigned char)*w, h, cudaMemcpyDeviceToHost));
}

void runConsistencyCheck(unsigned char *h_dc, unsigned char *d_left, unsigned char *d_right, size_t pitches, unsigned int w, unsigned int h, unsigned char threshold) {
	dim3 numThreads = dim3(CONSISTENCY_KERNEL_blockSize_x, CONSISTENCY_KERNEL_blockSize_y, 1);
	dim3 numBlocks = dim3(iDivUp(pitches, numThreads.x), iDivUp(h, numThreads.y));
	cudaTextureObject_t tex2Dleft, tex2Dright;
	init2DTextureObj(&tex2Dleft, TDT_U8, pitches, d_left, w, h);
	init2DTextureObj(&tex2Dright, TDT_U8, pitches, d_right, w, h);
	consistencyKernel << <numBlocks, numThreads >> > (d_left, tex2Dleft, tex2Dright, w, h, pitches, threshold);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy2D(h_dc, sizeof(unsigned char)*w, d_left, pitches, sizeof(unsigned char)*w, h, cudaMemcpyDeviceToHost));
}
