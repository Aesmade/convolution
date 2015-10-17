#include "convolution.h"

#ifdef _OPENMP
#include <omp.h>
#define THREADNUM 2
#endif

// shortcut for adding the 3 pixels in the previous row from OFFSET, multiplied by their coefficients
// if STREAMCOUNT is 1 then we add the 3 bytes directly above the byte at OFFSET (greyscale)
// otherwise we add every 3rd byte (RGB)
#define ADD_UPPER_ROW(OFFSET, STREAMCOUNT) src_buf[OFFSET-blockwidth-STREAMCOUNT] * filter[DIR_DRIGHT] + \
                            src_buf[OFFSET-blockwidth] * filter[DIR_DOWN] + \
                            src_buf[OFFSET-blockwidth+STREAMCOUNT] * filter[DIR_DLEFT];

// add the pixels at OFFSET, before OFFSET and after OFFSET
#define ADD_MIDDLE_ROW(OFFSET, STREAMCOUNT) src_buf[OFFSET] * filter[DIR_CENTER] + \
                            src_buf[OFFSET-STREAMCOUNT] * filter[DIR_RIGHT] + \
                            src_buf[OFFSET+STREAMCOUNT] * filter[DIR_LEFT];

// add the pixels in the row below OFFSET
#define ADD_DOWN_ROW(OFFSET, STREAMCOUNT) src_buf[OFFSET+blockwidth-STREAMCOUNT] * filter[DIR_URIGHT] + \
                            src_buf[OFFSET+blockwidth] * filter[DIR_UP] + \
                            src_buf[OFFSET+blockwidth+STREAMCOUNT] * filter[DIR_ULEFT];

// add the pixels at the column left of OFFSET
#define ADD_LEFT_COL(OFFSET, STREAMCOUNT) src_buf[OFFSET-blockwidth-STREAMCOUNT] * filter[DIR_DRIGHT] + \
                            src_buf[OFFSET-1] * filter[DIR_RIGHT] + \
                            src_buf[OFFSET+blockwidth-STREAMCOUNT] * filter[DIR_URIGHT];

// add the pixels at OFFSET, above it and below it
#define ADD_MIDDLE_COL(OFFSET, STREAMCOUNT) src_buf[OFFSET] * filter[DIR_CENTER] + \
                            src_buf[OFFSET-blockwidth] * filter[DIR_DOWN] + \
                            src_buf[OFFSET+blockwidth] * filter[DIR_UP];

// add the pixels at the column left of OFFSET
#define ADD_RIGHT_COL(OFFSET, STREAMCOUNT) src_buf[OFFSET-blockwidth+STREAMCOUNT] * filter[DIR_DLEFT] + \
                            src_buf[OFFSET+STREAMCOUNT] * filter[DIR_LEFT] + \
                            src_buf[OFFSET+blockwidth+STREAMCOUNT] * filter[DIR_ULEFT];


MPI_Datatype type_column;
MPI_Datatype type_column_rgb;

// prepare the column type
void prepare_coltype(int blockwidth, int blockheight) {
    MPI_Type_vector(blockheight, 1, blockwidth, MPI_CHAR,
    &type_column);
    MPI_Type_commit(&type_column);

    MPI_Type_vector(blockheight, 3, blockwidth, MPI_CHAR,
    &type_column_rgb);
    MPI_Type_commit(&type_column_rgb);
}

// store the ranks of neighbouring processes in ranks[]
void get_ranks(int cur_rank, MPI_Comm cartesian, int ranks[]) {
    int coords[2];
    MPI_Cart_coords(cartesian, cur_rank, 2, coords);
    ranks[DIR_CENTER] = cur_rank;
    coords[1]--;
    coords[0]--;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_ULEFT]);
    coords[1]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_UP]);
    coords[1]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_URIGHT]);
    coords[0]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_RIGHT]);
    coords[1] -= 2;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_LEFT]);
    coords[0]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_DLEFT]);
    coords[1]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_DOWN]);
    coords[1]++;
    MPI_Cart_rank(cartesian, coords, &ranks[DIR_DRIGHT]);
}

// process the image in src_buf, total_reps times
// the image is copied between src_buf and dest_buf while being processed
// returns the buffer the final version of the image is in
unsigned char *convolution(int blockwidth, int blockheight, unsigned char *filter, int ranks[], int total_reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf) {
    MPI_Request sendRequests[9], recvRequests[9];
    unsigned char ulCorner, urCorner, dlCorner, drCorner;
    // allocate buffers for receiving the columns of pixels at the left and right and the rows at the top and bottom
    unsigned char *upRow = (char*)malloc(blockwidth), *downRow = (char*)malloc(blockwidth),
    *leftCol = (char*)malloc(blockheight), *rightCol = (char*)malloc(blockheight);

    // get ranks of neighbouring processes
    int rUleft = ranks[DIR_ULEFT], rUp = ranks[DIR_UP], rUright = ranks[DIR_URIGHT], rRight = ranks[DIR_RIGHT],
    rLeft = ranks[DIR_LEFT], rDleft = ranks[DIR_DLEFT], rDown = ranks[DIR_DOWN], rDright = ranks[DIR_DRIGHT];

    // the sum of the filter values
    unsigned int filt_total = 0;
    for (int i = 0; i < 9; i++)
        filt_total += filter[i];

    for (int rep = 0; rep < total_reps; rep++) {
        // send needed pixels to other processes (top and bottom row, left and right column and the 4 corner pixels)
        MPI_Isend(&src_buf[0], 1, MPI_CHAR, rUleft, DIR_ULEFT, cartesian, &sendRequests[DIR_ULEFT]);
        MPI_Isend(src_buf, blockwidth, MPI_CHAR, rUp, DIR_UP, cartesian, &sendRequests[DIR_UP]);
        MPI_Isend(&src_buf[blockwidth-1], 1, MPI_CHAR, rUright, DIR_URIGHT, cartesian, &sendRequests[DIR_URIGHT]);
        MPI_Isend(src_buf, 1, type_column, rLeft, DIR_LEFT, cartesian, &sendRequests[DIR_LEFT]);
        MPI_Isend(&src_buf[blockwidth-1], 1, type_column, rRight, DIR_RIGHT, cartesian, &sendRequests[DIR_RIGHT]);
        MPI_Isend(&src_buf[(blockheight-1)*blockwidth], 1, MPI_CHAR, rDleft, DIR_DLEFT, cartesian, &sendRequests[DIR_DLEFT]);
        MPI_Isend(&src_buf[(blockheight-1)*blockwidth], blockwidth, MPI_CHAR, rDown, DIR_DOWN, cartesian, &sendRequests[DIR_DOWN]);
        MPI_Isend(&src_buf[blockheight*blockwidth-1], 1, MPI_CHAR, rDright, DIR_DRIGHT, cartesian, &sendRequests[DIR_DRIGHT]);

        sendRequests[DIR_CENTER] = MPI_REQUEST_NULL;

        // get needed pixels from other processes
        MPI_Irecv(&ulCorner, 1, MPI_CHAR, rUleft, DIR_DRIGHT, cartesian, &recvRequests[DIR_ULEFT]);
        MPI_Irecv(upRow, blockwidth, MPI_CHAR, rUp, DIR_DOWN, cartesian, &recvRequests[DIR_UP]);
        MPI_Irecv(&urCorner, 1, MPI_CHAR, rUright, DIR_DLEFT, cartesian, &recvRequests[DIR_URIGHT]);
        MPI_Irecv(leftCol, blockheight, MPI_CHAR, rLeft, DIR_RIGHT, cartesian, &recvRequests[DIR_LEFT]);
        MPI_Irecv(rightCol, blockheight, MPI_CHAR, rRight, DIR_LEFT, cartesian, &recvRequests[DIR_RIGHT]);
        MPI_Irecv(&dlCorner, 1, MPI_CHAR, rDleft, DIR_URIGHT, cartesian, &recvRequests[DIR_DLEFT]);
        MPI_Irecv(downRow, blockwidth, MPI_CHAR, rDown, DIR_UP, cartesian, &recvRequests[DIR_DOWN]);
        MPI_Irecv(&drCorner, 1, MPI_CHAR, rDright, DIR_ULEFT, cartesian, &recvRequests[DIR_DRIGHT]);

        recvRequests[DIR_CENTER] = MPI_REQUEST_NULL;

        // calculate inside pixels
        // if using OpenMP, use a different thread for each row
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(THREADNUM)
        #endif
        for (int i = 1; i < blockheight-1; i++)
            for (int l = 1; l < blockwidth-1; l++) {
                // find offset of current pixel in array
                unsigned int offset = i*blockwidth + l;
                // for inside pixels, just add all 9 pixels multiplied by their coefficients
                unsigned int val = ADD_UPPER_ROW(offset, 1);
                val += ADD_MIDDLE_ROW(offset, 1);
                val += ADD_DOWN_ROW(offset, 1);

                // divide by total and write to output
                val /= filt_total;
                dest_buf[offset] = (char)val & 0xFF;
            }

        // wait until we receive all needed pixels for pixels on the edges
        MPI_Waitall(8, recvRequests, MPI_STATUSES_IGNORE);

        // calculate pixels on top edge
        for (int i = 1; i < blockwidth-1; i++) {
            // sum of pixels inside the block
            unsigned int val = ADD_MIDDLE_ROW(i, 1);
            val += ADD_DOWN_ROW(i, 1);
            // add pixels from the block above the current one
            val += upRow[i-1] * filter[DIR_DRIGHT];
            val += upRow[i] * filter[DIR_DOWN];
            val += upRow[i+1] * filter[DIR_DLEFT];

            val /= filt_total;
            dest_buf[i] = (char)val & 0xFF;
        }

        // calculate pixels on bottom edge
        unsigned int last_line = (blockheight-1)*blockwidth;
        for (int i = 1; i < blockwidth-1; i++) {
            unsigned int val = ADD_MIDDLE_ROW(last_line+i, 1);
            val += ADD_UPPER_ROW(last_line+i, 1);

            val += downRow[i-1] * filter[DIR_URIGHT];
            val += downRow[i] * filter[DIR_UP];
            val += downRow[i+1] * filter[DIR_ULEFT];

            val /= filt_total;
            dest_buf[last_line+i] = (char)val & 0xFF;
        }

        // calculate pixels on left edge
        for (int i = 1; i < blockheight-1; i++) {
            unsigned int base = i*blockwidth;
            unsigned int val = ADD_MIDDLE_COL(base, 1);
            val += ADD_RIGHT_COL(base, 1);

            val += leftCol[i-1] * filter[DIR_DRIGHT];
            val += leftCol[i] * filter[DIR_RIGHT];
            val += leftCol[i+1] * filter[DIR_URIGHT];

            val /= filt_total;
            dest_buf[base] = (char)val & 0xFF;
        }

        // calculate pixels on right edge
        for (int i = 1; i < blockheight-1; i++) {
            unsigned int base = (i+1)*blockwidth - 1;
            unsigned int val = ADD_MIDDLE_COL(base, 1);
            val += ADD_LEFT_COL(base, 1);

            val += rightCol[i-1] * filter[DIR_DLEFT];
            val += rightCol[i] * filter[DIR_LEFT];
            val += rightCol[i+1] * filter[DIR_ULEFT];

            val /= filt_total;
            dest_buf[base] = (char)val & 0xFF;
        }

        unsigned int base = 0;
        unsigned int val = 0;

        // calculate corners

        val = src_buf[0] * filter[DIR_CENTER];
        val += src_buf[1] * filter[DIR_LEFT];
        val += src_buf[blockwidth] * filter[DIR_UP];
        val += src_buf[blockwidth+1] * filter[DIR_ULEFT];
        val += upRow[0] * filter[DIR_DOWN];
        val += upRow[1] * filter[DIR_DLEFT];
        val += leftCol[0] * filter[DIR_RIGHT];
        val += leftCol[1] * filter[DIR_URIGHT];
        val += ulCorner * filter[DIR_DRIGHT];

        val /= filt_total;
        dest_buf[0] = (char)val & 0xFF;

        base = blockwidth-1;
        val = src_buf[base] * filter[DIR_CENTER];
        val += src_buf[base-1] * filter[DIR_RIGHT];
        val += src_buf[base+blockwidth] * filter[DIR_UP];
        val += src_buf[base+blockwidth-1] * filter[DIR_URIGHT];
        val += upRow[base] * filter[DIR_DOWN];
        val += upRow[base-1] * filter[DIR_DRIGHT];
        val += rightCol[0] * filter[DIR_LEFT];
        val += rightCol[1] * filter[DIR_ULEFT];
        val += urCorner * filter[DIR_DLEFT];

        val /= filt_total;
        dest_buf[base] = (char)val & 0xFF;

        base = (blockheight-1)*blockwidth;
        val = src_buf[base] * filter[DIR_CENTER];
        val += src_buf[base+1] * filter[DIR_LEFT];
        val += src_buf[base-blockwidth] * filter[DIR_DOWN];
        val += src_buf[base-blockwidth+1] * filter[DIR_DLEFT];
        val += downRow[0] * filter[DIR_UP];
        val += downRow[1] * filter[DIR_ULEFT];
        val += leftCol[blockheight-1] * filter[DIR_RIGHT];
        val += leftCol[blockheight-2] * filter[DIR_DRIGHT];
        val += dlCorner * filter[DIR_URIGHT];

        val /= filt_total;
        dest_buf[base] = (char)val & 0xFF;

        base = blockheight*blockwidth-1;
        val = src_buf[base] * filter[DIR_CENTER];
        val += src_buf[base-1] * filter[DIR_RIGHT];
        val += src_buf[base-blockwidth] * filter[DIR_DOWN];
        val += src_buf[base-blockwidth-1] * filter[DIR_DRIGHT];
        val += downRow[blockwidth-1] * filter[DIR_UP];
        val += downRow[blockwidth-2] * filter[DIR_URIGHT];
        val += rightCol[blockheight-1] * filter[DIR_LEFT];
        val += rightCol[blockheight-2] * filter[DIR_DLEFT];
        val += drCorner * filter[DIR_ULEFT];

        val /= filt_total;
        dest_buf[base] = (char)val & 0xFF;

        // wait for all sends to process
        MPI_Waitall(8, sendRequests, MPI_STATUSES_IGNORE);

        // swap src and dest buffers for the next round
        unsigned char *tmp = dest_buf;
        dest_buf = src_buf;
        src_buf = tmp;

        // if last 5 bits of rep are ones (every 32 rounds)
        if ((rep & 31) == 31) {
            unsigned char any_diff, diff = 0;
            // find if last round made any difference, send 1 if it did or 0 if it didn't
            for (int i = 0; i < blockwidth * blockheight; i++)
                if (src_buf[i] != dest_buf[i]) {
                    diff = 1;
                    break;
                }
            // send to root, which finds the max value sent and broadcasts it
            MPI_Reduce(&diff, &any_diff, 1, MPI_CHAR, MPI_MAX, 0, cartesian);
            MPI_Bcast(&any_diff, 1, MPI_CHAR, 0, cartesian);

            // if the max value is 0, all threads sent 0 so the image has converged so we can stop
            if (any_diff == 0)
                break;
        }
    }

    free(upRow);
    free(leftCol);
    free(rightCol);
    free(downRow);

    return src_buf;
}

// same as above, but for 3 streams
unsigned char *convolution_rgb(int blockwidth, int blockheight, unsigned char *filter, int ranks[], int total_reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf) {
    MPI_Request sendRequests[9], recvRequests[9];
    unsigned char ulCorner[3], urCorner[3], dlCorner[3], drCorner[3];
    unsigned char *upRow = (char*)malloc(blockwidth), *downRow = (char*)malloc(blockwidth),
    *leftCol = (char*)malloc(blockheight*3), *rightCol = (char*)malloc(blockheight*3);

    int rUleft = ranks[DIR_ULEFT], rUp = ranks[DIR_UP], rUright = ranks[DIR_URIGHT], rRight = ranks[DIR_RIGHT],
    rLeft = ranks[DIR_LEFT], rDleft = ranks[DIR_DLEFT], rDown = ranks[DIR_DOWN], rDright = ranks[DIR_DRIGHT];

    unsigned int filt_total = 0;
    for (int i = 0; i < 9; i++)
        filt_total += filter[i];

    for (int rep = 0; rep < total_reps; rep++) {
        MPI_Isend(&src_buf[0], 3, MPI_CHAR, rUleft, DIR_ULEFT, cartesian, &sendRequests[DIR_ULEFT]);
        MPI_Isend(src_buf, blockwidth, MPI_CHAR, rUp, DIR_UP, cartesian, &sendRequests[DIR_UP]);
        MPI_Isend(&src_buf[blockwidth-3], 3, MPI_CHAR, rUright, DIR_URIGHT, cartesian, &sendRequests[DIR_URIGHT]);
        MPI_Isend(src_buf, 1, type_column_rgb, rLeft, DIR_LEFT, cartesian, &sendRequests[DIR_LEFT]);
        MPI_Isend(&src_buf[blockwidth-3], 1, type_column_rgb, rRight, DIR_RIGHT, cartesian, &sendRequests[DIR_RIGHT]);
        MPI_Isend(&src_buf[(blockheight-1)*blockwidth], 3, MPI_CHAR, rDleft, DIR_DLEFT, cartesian, &sendRequests[DIR_DLEFT]);
        MPI_Isend(&src_buf[(blockheight-1)*blockwidth], blockwidth, MPI_CHAR, rDown, DIR_DOWN, cartesian, &sendRequests[DIR_DOWN]);
        MPI_Isend(&src_buf[blockheight*blockwidth-3], 3, MPI_CHAR, rDright, DIR_DRIGHT, cartesian, &sendRequests[DIR_DRIGHT]);

        sendRequests[DIR_CENTER] = MPI_REQUEST_NULL;

        MPI_Irecv(&ulCorner, 3, MPI_CHAR, rUleft, DIR_DRIGHT, cartesian, &recvRequests[DIR_ULEFT]);
        MPI_Irecv(upRow, blockwidth, MPI_CHAR, rUp, DIR_DOWN, cartesian, &recvRequests[DIR_UP]);
        MPI_Irecv(&urCorner, 3, MPI_CHAR, rUright, DIR_DLEFT, cartesian, &recvRequests[DIR_URIGHT]);
        MPI_Irecv(leftCol, blockheight*3, MPI_CHAR, rLeft, DIR_RIGHT, cartesian, &recvRequests[DIR_LEFT]);
        MPI_Irecv(rightCol, blockheight*3, MPI_CHAR, rRight, DIR_LEFT, cartesian, &recvRequests[DIR_RIGHT]);
        MPI_Irecv(&dlCorner, 3, MPI_CHAR, rDleft, DIR_URIGHT, cartesian, &recvRequests[DIR_DLEFT]);
        MPI_Irecv(downRow, blockwidth, MPI_CHAR, rDown, DIR_UP, cartesian, &recvRequests[DIR_DOWN]);
        MPI_Irecv(&drCorner, 3, MPI_CHAR, rDright, DIR_ULEFT, cartesian, &recvRequests[DIR_DRIGHT]);

        recvRequests[DIR_CENTER] = MPI_REQUEST_NULL;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(THREADNUM)
        #endif

        // for inside pixels
        for (int i = 1; i < blockheight-1; i++)
            for (int l = 3; l < blockwidth-3; l ++) {

                // add rows similar to greyscale, but add every 3 bytes instead of 1
                unsigned int offset = i*blockwidth + l;
                unsigned int val = ADD_UPPER_ROW(offset, 3);
                val += ADD_MIDDLE_ROW(offset, 3);
                val += ADD_DOWN_ROW(offset, 3);

                val /= filt_total;
                dest_buf[offset] = (char)val & 0xFF;
            }

        MPI_Waitall(8, recvRequests, MPI_STATUSES_IGNORE);

        for (int i = 3; i < blockwidth-3; i++) {
            unsigned int val = ADD_MIDDLE_ROW(i, 3);
            val += ADD_DOWN_ROW(i, 3);

            val += upRow[i-3] * filter[DIR_DRIGHT];
            val += upRow[i] * filter[DIR_DOWN];
            val += upRow[i+3] * filter[DIR_DLEFT];

            val /= filt_total;
            dest_buf[i] = (char)val & 0xFF;
        }

        unsigned int last_line = (blockheight-1)*blockwidth;
        for (int i = 3; i < blockwidth-3; i++) {
            unsigned int val = ADD_UPPER_ROW(last_line+i, 3);
            val += ADD_MIDDLE_ROW(last_line+i, 3);

            val += downRow[i-3] * filter[DIR_URIGHT];
            val += downRow[i] * filter[DIR_UP];
            val += downRow[i+3] * filter[DIR_ULEFT];

            val /= filt_total;
            dest_buf[last_line+i] = (char)val & 0xFF;
        }

        unsigned int base = 0;
        unsigned int val = 0;

        // process corners and left and right edges for each stream
        for (unsigned char stream = 0; stream < 3; stream++) {
        	for (unsigned int i = 1; i < blockheight-1; i++) {
	            base = i*blockwidth + stream;
	            val = ADD_MIDDLE_COL(base, 3);
                val += ADD_RIGHT_COL(base, 3);

	            val += leftCol[(i-1)*3 + stream] * filter[DIR_DRIGHT];
	            val += leftCol[i*3 + stream] * filter[DIR_RIGHT];
	            val += leftCol[(i+1)*3 + stream] * filter[DIR_URIGHT];

	            val /= filt_total;
	            dest_buf[base] = (char)val & 0xFF;
	        }

	        for (unsigned int i = 1; i < blockheight-1; i++) {
	            base = (i+1)*blockwidth - 3 + stream;
	            val = ADD_MIDDLE_COL(base, 3);
                val += ADD_LEFT_COL(base, 3);

	            val += rightCol[(i-1)*3 + stream] * filter[DIR_DLEFT];
	            val += rightCol[i*3 + stream] * filter[DIR_LEFT];
	            val += rightCol[(i+1)*3 + stream] * filter[DIR_ULEFT];

	            val /= filt_total;
	            dest_buf[base] = (char)val & 0xFF;
	        }

            val = src_buf[stream] * filter[DIR_CENTER];
            val += src_buf[stream+3] * filter[DIR_LEFT];
            val += src_buf[stream+blockwidth] * filter[DIR_UP];
            val += src_buf[stream+blockwidth+3] * filter[DIR_ULEFT];
            val += upRow[stream] * filter[DIR_DOWN];
            val += upRow[stream+3] * filter[DIR_DLEFT];
            val += leftCol[stream] * filter[DIR_RIGHT];
            val += leftCol[stream+3] * filter[DIR_URIGHT];
            val += ulCorner[stream] * filter[DIR_DRIGHT];
            val /= filt_total;
            dest_buf[stream] = (char)val & 0xFF;

            base = blockwidth+stream-3;
            val = src_buf[base] * filter[DIR_CENTER];
            val += src_buf[base-3] * filter[DIR_RIGHT];
            val += src_buf[base+blockwidth] * filter[DIR_UP];
            val += src_buf[base+blockwidth-3] * filter[DIR_URIGHT];
            val += upRow[base] * filter[DIR_DOWN];
            val += upRow[base-3] * filter[DIR_DRIGHT];
            val += rightCol[stream] * filter[DIR_LEFT];
            val += rightCol[stream+3] * filter[DIR_ULEFT];
            val += urCorner[stream] * filter[DIR_DLEFT];
            val /= filt_total;
            dest_buf[base] = (char)val & 0xFF;

            base = (blockheight-1)*blockwidth+stream;
            val = src_buf[base] * filter[DIR_CENTER];
            val += src_buf[base+3] * filter[DIR_LEFT];
            val += src_buf[base-blockwidth] * filter[DIR_DOWN];
            val += src_buf[base-blockwidth+3] * filter[DIR_DLEFT];
            val += downRow[stream] * filter[DIR_UP];
            val += downRow[stream+3] * filter[DIR_ULEFT];
            val += leftCol[blockheight+stream-3] * filter[DIR_RIGHT];
            val += leftCol[blockheight+stream-6] * filter[DIR_DRIGHT];
            val += dlCorner[stream] * filter[DIR_URIGHT];
            val /= filt_total;
            dest_buf[base] = (char)val & 0xFF;

            base = blockheight*blockwidth+stream-3;
            val = src_buf[base] * filter[DIR_CENTER];
            val += src_buf[base-3] * filter[DIR_RIGHT];
            val += src_buf[base-blockwidth] * filter[DIR_DOWN];
            val += src_buf[base-blockwidth-3] * filter[DIR_DRIGHT];
            val += downRow[blockwidth+stream-3] * filter[DIR_UP];
            val += downRow[blockwidth+stream-6] * filter[DIR_URIGHT];
            val += rightCol[blockheight+stream-3] * filter[DIR_LEFT];
            val += rightCol[blockheight+stream-6] * filter[DIR_DLEFT];
            val += drCorner[stream] * filter[DIR_ULEFT];
            val /= filt_total;
            dest_buf[base] = (char)val & 0xFF;
        }

        MPI_Waitall(8, sendRequests, MPI_STATUSES_IGNORE);

        unsigned char *tmp = dest_buf;
        dest_buf = src_buf;
        src_buf = tmp;

        if ((rep & 31) == 31) {
            unsigned char any_diff, diff = 0;
            for (int i = 0; i < blockwidth * blockheight; i++)
                if (src_buf[i] != dest_buf[i]) {
                    diff = 1;
                    break;
                }
            MPI_Reduce(&diff, &any_diff, 1, MPI_CHAR, MPI_MAX, 0, cartesian);
            MPI_Bcast(&any_diff, 1, MPI_CHAR, 0, cartesian);

            if (any_diff == 0)
                break;
        }
    }

    free(upRow);
    free(leftCol);
    free(rightCol);
    free(downRow);

    return src_buf;
}