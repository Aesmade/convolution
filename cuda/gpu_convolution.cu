#include <lcutil.h>

#define FILTER(v,x,y) v*filter[-y*3-x+4]

__global__ void cuda_conv(unsigned int width, unsigned int height, unsigned short *filter, unsigned int total, unsigned char *src, unsigned char *dest) {
    // find current thread x and y
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        unsigned int offset = x + y * width;
        unsigned int res = FILTER(src[offset], 0, 0);
        unsigned int left = -1, right = 1, up = -width, down = width;
        // if we are on the left edge, offset to go left should be 0 as we can't go left
        if (x == 0)
            left = 0;
        // similar for other directions
        else if (x == width - 1)
            right = 0;
        if (y == 0)
            up = 0;
        else if (y == height - 1)
            down = 0;
        // multiply neighbouring pixels by their coefficients and add to result
        res += FILTER(src[offset+right+down], -1, -1);
        res += FILTER(src[offset+down], 0, -1);
        res += FILTER(src[offset+left+down], 1, -1);
        res += FILTER(src[offset+right], -1, 0);
        res += FILTER(src[offset+left], 1, 0);
        res += FILTER(src[offset+right+up], -1, 1);
        res += FILTER(src[offset+up], 0, 1);
        res += FILTER(src[offset+left+up], 1, 1);
        // divide result by the sum of all coefficients and write to target
        dest[offset] = (unsigned char)((res / total) & 0xFF);
    }
}

__global__ void cuda_conv_rgb(unsigned int width, unsigned int height, unsigned short *filter, unsigned int total, unsigned char *src, unsigned char *dest) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        // same as above, but calculate each channel separately
        unsigned int offset = (x + y * width) * 3;
        unsigned int left = -3, right = 3, up = -3*width, down = 3*width;
        if (x == 0)
            left = 0;
        else if (x == width - 1)
            right = 0;
        if (y == 0)
            up = 0;
        else if (y == height - 1)
            down = 0;

        for (int i = 0; i < 3; i++) {
            unsigned int res = FILTER(src[offset], 0, 0);
            res += FILTER(src[offset+right+down], -1, -1);
            res += FILTER(src[offset+down], 0, -1);
            res += FILTER(src[offset+left+down], 1, -1);
            res += FILTER(src[offset+right], -1, 0);
            res += FILTER(src[offset+left], 1, 0);
            res += FILTER(src[offset+right+up], -1, 1);
            res += FILTER(src[offset+up], 0, 1);
            res += FILTER(src[offset+left+up], 1, 1);
            dest[offset] = (unsigned char)((res / total) & 0xFF);

            offset++;
        }
    }
}

__global__ void cuda_memcmp(unsigned char *buf1, unsigned char *buf2, unsigned int len, char *res) {
    // if we find any difference, write 0 to output and return
    for (unsigned int i = 0; i < len; i++)
        if (buf1[i] != buf2[i]) {
            *res = 0;
            return;
        }
    // otherwise write 1
    *res = 1;
}

extern "C" float convolutionGPU(unsigned int width, unsigned int height, unsigned short *filter, unsigned int is_rgb, unsigned int rounds, unsigned int bufsize, unsigned char *buffer){
    unsigned char *dev_buf1, *dev_buf2;
    unsigned short *dev_filter;
    // allocate needed device buffers
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_buf1, bufsize) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_buf2, bufsize) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_filter, 9*sizeof(unsigned short)) );

    // copy data to device memory
    CUDA_SAFE_CALL( cudaMemcpy(dev_buf1, buffer, bufsize, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(dev_filter, filter, 9*sizeof(unsigned short), cudaMemcpyHostToDevice) );

    dim3 dimBl(24, 32);
    dim3 dimGr(FRACTION_CEILING(width, 24), FRACTION_CEILING(height, 32));
    unsigned int total = 0;
    for (int i = 0; i < 9; i++)
        total += filter[i];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    char converge = 0;
    char *dev_converge;
    // allocate one byte on the device as a flag for convergence
    CUDA_SAFE_CALL( cudaMalloc(&dev_converge, 1) );
    // start with convergence = 0
    CUDA_SAFE_CALL( cudaMemcpy(dev_converge, &converge, 1, cudaMemcpyHostToDevice) );
    if (is_rgb == 0) {
        // run the convolution function twice, once with buf1 as the source
        // and buf2 as the target and once in reverse
        for (int i = 0; i < rounds / 2; i++) {
            cuda_conv<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf1, dev_buf2);
            cuda_conv<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf2, dev_buf1);
            if ((i & 15) == 15) {
                // every 16 loops (32 rounds), compare the two buffers
                cuda_memcmp<<<1, 1>>>(dev_buf1, dev_buf2, bufsize, dev_converge);
                // copy the flag from the device to the host
                CUDA_SAFE_CALL( cudaMemcpy(&converge, dev_converge, 1, cudaMemcpyDeviceToHost) );
                // if no change between them, break
                if (converge == 1)
                    break;
            }
        }
        // if the last bit is 1, we need to run it one more time and move the result to dev_buf1
        if (converge == 0 && (rounds & 1) == 1) {
            cuda_conv<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf1, dev_buf2);
            unsigned char *tmp = dev_buf1;
            dev_buf1 = dev_buf2;
            dev_buf2 = tmp;
        }
    } else {
        // same as above but for RGB
        for (int i = 0; i < rounds / 2; i++) {
            cuda_conv_rgb<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf1, dev_buf2);
            cuda_conv_rgb<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf2, dev_buf1);
            if ((i & 15) == 15) {
                cuda_memcmp<<<1, 1>>>(dev_buf1, dev_buf2, bufsize, dev_converge);
                CUDA_SAFE_CALL( cudaMemcpy(&converge, dev_converge, 1, cudaMemcpyDeviceToHost) );
                if (converge == 1)
                    break;
            }
        }
        if (converge == 0 && (rounds & 1) == 1) {
            cuda_conv_rgb<<<dimGr, dimBl>>>(width, height, dev_filter, total, dev_buf1, dev_buf2);
            unsigned char *tmp = dev_buf1;
            dev_buf1 = dev_buf2;
            dev_buf2 = tmp;
        }
    }
    CUDA_SAFE_CALL( cudaFree(dev_converge) );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CUDA_SAFE_CALL( cudaGetLastError() );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // copy results back to host memory
    CUDA_SAFE_CALL( cudaMemcpy(buffer, dev_buf1, bufsize, cudaMemcpyDeviceToHost) );

    CUDA_SAFE_CALL( cudaFree(dev_buf1) );
    CUDA_SAFE_CALL( cudaFree(dev_buf2) );
    CUDA_SAFE_CALL( cudaFree(dev_filter) );
    return milliseconds;
}

