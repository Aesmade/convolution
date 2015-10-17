#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" float convolutionGPU(unsigned int,unsigned int,unsigned short*,unsigned int,unsigned int,unsigned int,unsigned char*);

int main (int argc, char **argv) {
    char c;
    unsigned int width = 1920;
    unsigned int height = 2520;
    int is_rgb = 0;
    unsigned int rounds = -1;
    char *fname = (char*)"waterfall_grey_1920_2520.raw", *filtstr = (char*)"1,2,1,2,4,2,1,2,1";

    printf("Options:\n"
        "\t-f file\t\tSet the name of the image file\n"
        "\t-w width\tSet the width of the image\n"
        "\t-h height\tSet the height of the image\n"
        "\t-c\t\tSpecify an RGB image\n"
        "\t-v filter\tNine comma separated coefficients for the convolution filter (left to right, up to down)\n"
        "\t-r rounds\tNumber of rounds to run the algorithm (will run until convergence if not specified)\n"
    );

    // parse parameters
    opterr = 0;
    while ((c = getopt (argc, argv, "f:w:h:cv:r:")) != -1)
        switch (c) {
        case 'f':
        fname = optarg;
        break;
        case 'w':
        width = atoi(optarg);
        break;
        case 'h':
        height = atoi(optarg);
        break;
        case 'c':
        is_rgb = 1;
        break;
        case 'v':
        filtstr = optarg;
        break;
        case 'r':
        rounds = atoi(optarg);
        break;
        case '?':
        if (isprint (optopt))
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
            fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
        default:
        abort ();
    }

    printf("Running with parameters:\nfilename = %s, width = %d, height = %d, RGB = %s, filter = %s, rounds = %d\n",
        fname, width, height, is_rgb == 0 ? "no" : "yes", filtstr, rounds);

    // calculate file size
    unsigned int fsize = width * height;
    if (is_rgb == 1)
        fsize *= 3;
    unsigned char *file_buf = (unsigned char*) malloc(fsize);

    FILE *f_input = fopen(fname, "rb");
    unsigned int bytes_read = 0;
    while (bytes_read < fsize)
        bytes_read += fread(&file_buf[bytes_read], 1, fsize - bytes_read, f_input);
    fclose(f_input);

    unsigned short flt[9];

    // read filter from the string
    char *filtstr_dup = strdup(filtstr);
    char *token = strtok(filtstr_dup, ",");
    if (token != NULL) {
        for (int i = 0; i < 9 && token != NULL; i++) {
            flt[i] = atoi(token);
            token = strtok(NULL, ",");
        }
    }
    free(filtstr_dup);

    // run the CUDA code and print the time
    printf("Elapsed time: %f\n", convolutionGPU(width, height, flt, is_rgb, rounds, fsize, file_buf));

    FILE *f_out = fopen("out.raw", "wb");
    fwrite(file_buf, 1, fsize, f_out);
    fclose(f_out);

    free(file_buf);

    return 0;
}