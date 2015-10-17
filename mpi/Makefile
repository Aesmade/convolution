OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
MPICC = mpicc
CLEAN_FILES = mpi_conv omp_conv main.o convolution.o convolution_omp.o
FLAGS = -std=c99 -lm
OMPFLAGS = -fopenmp

all: mpi_conv omp_conv

mpi_conv: main.o convolution.o
	${MPICC} -o $@ $^ ${FLAGS}

omp_conv: main.o convolution_omp.o
	${MPICC} ${OMPFLAGS} -o $@ $^ ${FLAGS}

main.o: main.c
	${MPICC} -c ${FLAGS} -o $@ $^

convolution.o: convolution.c
	${MPICC} -c ${FLAGS} -o $@ $^

convolution_omp.o: convolution.c
	${MPICC} -c ${FLAGS} ${OMPFLAGS} -o $@ $^

clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/conv

rebuild: clean all

