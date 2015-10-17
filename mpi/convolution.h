#include <mpi.h>
#include <stdlib.h>

typedef enum {DIR_ULEFT = 0, DIR_UP, DIR_URIGHT, DIR_LEFT, DIR_CENTER,
    DIR_RIGHT, DIR_DLEFT, DIR_DOWN, DIR_DRIGHT} dir;

void prepare_coltype(int blockwidth, int blockheight);

void get_ranks(int cur_rank, MPI_Comm cartesian, int ranks[]);

unsigned char *convolution(int blockwidth, int blockheight, unsigned char *filter, int ranks[], int total_reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf);

unsigned char *convolution_rgb(int blockwidth, int blockheight, unsigned char *filter, int ranks[], int total_reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf);