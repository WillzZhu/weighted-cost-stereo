
#ifndef _GREYSCALE_KERNEL_H_
#define _GREYSCALE_KERNEL_H_

#define GREYSCALE_KERNEL_blockSize_x 32
#define GREYSCALE_KERNEL_blockSize_y 8

__global__ void greyscaleKernel(unsigned int *g_img0, unsigned int *g_img1,
								unsigned char *g_odata0, unsigned char *g_odata1,
								size_t pitches_out, 
								cudaTextureObject_t tex2Dleft,
								cudaTextureObject_t tex2Dright)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int left_pixel = tex2D<unsigned int>(tex2Dleft, tidx, tidy);
	unsigned int right_pixel = tex2D<unsigned int>(tex2Dright, tidx, tidy);
	unsigned char rl, rr, gl, gr, bl, br;
	rl = (0x000000ff & left_pixel); rr = (0x000000ff & right_pixel);
	gl = (0x0000ff00 & left_pixel) >> 8; gr = (0x0000ff00 & right_pixel) >> 8;
	bl = (0x00ff0000 & left_pixel) >> 16; br = (0x00ff0000 & right_pixel) >> 16;
	g_odata0[tidy * pitches_out + tidx] = (unsigned char)(0.2989 * (double)(rl)+0.5870 * (double)(gl)+0.1140 * (double)(bl));
	g_odata1[tidy * pitches_out + tidx] = (unsigned char)(0.2989 * (double)(rr)+0.5870 * (double)(gr)+0.1140 * (double)(br));
}

void cpu_greyscale(unsigned char *h_img, unsigned char *h_odata, unsigned int w, unsigned int h) {
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			unsigned char *r = (h_img + (i*w + j) * 4);
			unsigned char *g = (h_img + (i*w + j) * 4 + 1);
			unsigned char *b = (h_img + (i*w + j) * 4 + 2);
			*(h_odata + (i*w + j)) = (unsigned char)(0.2989 * (double)(*r) + 0.5870 * (double)(*g) + 0.1140 * (double)(*b));
		}
	}
}
#endif