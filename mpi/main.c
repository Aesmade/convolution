#include "convolution.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Comm cartesian;
    double starttime, endtime, elapsed, elapsed_max;
    int size, rank;
    unsigned char filter[9] = {1,2,1,2,4,2,1,2,1};

    if (argc != 6) {
        printf("Usage: %s file width height is_rgb repetitions\n"
        "\tfile: File name of image\n"
        "\twidth: Width in pixels\n"
        "\theight: Height in pixels\n"
        "\nrgb: 1 for rgb, 0 for grey\n"
        "\trepetitions: Number of repetitions\n", argv[0]);
        return 0;
    }

    // read parameters
    char *filename = argv[1];
    unsigned int width = atoi(argv[2]);
    unsigned int height = atoi(argv[3]);
    unsigned int is_rgb = argv[4][0] == '1';
    unsigned int repetitions = atoi(argv[5]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (is_rgb == 1)
        width *= 3;

    int side = (int)sqrt(size);
    int blockwidth = width / side, blockheight = height / side;

    int sides[2] = {side, side};
    int periodic[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, sides, periodic, 1, &cartesian);
    if (cartesian == MPI_COMM_NULL) {
        printf("Could not create cartesian communicator\n");
        MPI_Finalize();
        return 1;
    }

    // set displacements and counts to use with MPI_Scatterv to send blocks of the image to the processes
    int *disps = malloc(size*sizeof(int));
    int *counts = malloc(size*sizeof(int));
    for (int ii=0; ii<side; ii++) {
        for (int jj=0; jj<side; jj++) {
            disps[ii*side+jj] = ii*width*blockheight+jj*blockwidth;
            counts [ii*side+jj] = 1;
        }
    }

    MPI_Datatype mpi_type_resized;
    MPI_Datatype mpi_type;

    // create datatype for sending blocks
    MPI_Type_vector(blockheight, blockwidth, width, MPI_CHAR, &mpi_type);
    MPI_Type_create_resized( mpi_type, 0, sizeof(char), &mpi_type_resized);
    MPI_Type_commit(&mpi_type_resized);

    unsigned char *buf = NULL;
    if (rank == 0) {
        buf = (char*)malloc(width*height);
        FILE *f = fopen(filename, "rb");
        fread(buf, 1, width*height, f);
        fclose(f);
    }

    // create buffers used to process the image
    unsigned char *local_buf = (char*)malloc(width*height/size), *local_buf2 = (char*)malloc(width*height/size);
    // scatter blocks to processes
    MPI_Scatterv(buf, counts, disps, mpi_type_resized, local_buf, width*height/size, MPI_CHAR, 0, cartesian);

    // prepare the column types used in the convolution
    prepare_coltype(blockwidth, blockheight);

    // save the ranks of each process' neighbours
    int neighbour_ranks[9];
    get_ranks(rank, cartesian, neighbour_ranks);

    MPI_Barrier(cartesian);
    starttime = MPI_Wtime();

    char *res_buf;

    // process image
    if (is_rgb == 0)
        res_buf = convolution(blockwidth, blockheight, (unsigned char*)filter, neighbour_ranks, repetitions, cartesian, local_buf, local_buf2);
    else
        res_buf = convolution_rgb(blockwidth, blockheight, (unsigned char*)filter, neighbour_ranks, repetitions, cartesian, local_buf, local_buf2);

    endtime = MPI_Wtime();
    elapsed = endtime - starttime;

    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);

    MPI_Gatherv(res_buf, width*height/size, MPI_CHAR, buf, counts, disps, mpi_type_resized, 0, cartesian);

    MPI_Finalize();
    if (rank == 0) {
        printf("Elapsed time: %f\n", elapsed_max);
        FILE *f = fopen("result.raw", "wb");
        fwrite(buf, 1, width*height, f);
        fclose(f);
        free(buf);
    }
    free(local_buf);
    free(local_buf2);
    free(disps);
    free(counts);
    return 0;
}