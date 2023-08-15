#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void allocate1Darray(int n, double * vec) {
	vec = (double*)malloc(n*sizeof(double));
}

void allocate2Darray(int n, double ** array) {
	int i;

	array = (double**)malloc(n*sizeof(double*));
	for (i = 0; i < n; i++) {
		array[i] = (double*)malloc(n*sizeof(double));
	}
}

void deallocate2Darray(int n, double ** array) {
	int i;

	for(int i = 0; i < n; i++) {
		free(array[i]);
	}
	free(array);
}






int main(int argc, char * argv[]) {
	int n = 100;
	int rank, nprocs;
	double **A, *b;

	allocate1Darray(n, b);
	allocate2Darray(n, A);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);









	MPI_Finalize();
	return 0;
}