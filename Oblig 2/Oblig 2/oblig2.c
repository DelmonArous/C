/* I denne obligen har jeg samarbeidet med Saurav Sharma */

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "mpi-utils.h"
#include <omp.h>

void read_matrix_binaryformat(char* filename, double*** matrix, int* num_rows, int* num_cols) {
	int i;
	FILE* fp = fopen(filename,"rb");
	fread(num_rows, sizeof(int), 1, fp);
	fread(num_cols, sizeof(int), 1, fp);

	/* storage allocation of the matrix */
	*matrix = (double**)malloc((*num_rows)*sizeof(double*));
	(*matrix)[0] = (double*)malloc((*num_rows)*(*num_cols)*sizeof(double));
	for (i = 1; i < (*num_rows); i++)
		(*matrix)[i] = (*matrix)[i-1] + (*num_cols);

	/* read in the entire matrix */
	fread((*matrix)[0], sizeof(double), (*num_rows)*(*num_cols), fp);
	fclose(fp);
}

void write_matrix_binaryformat(char* filename, double** matrix, int num_rows, int num_cols) {
	FILE *fp = fopen (filename,"wb");
	FILE *fr = fopen ("file.txt", "w");
	int m, n;

	fwrite(&num_rows, sizeof(int), 1, fp);
	fwrite(&num_cols, sizeof(int), 1, fp);
	fwrite(matrix[0], sizeof(double), num_rows*num_cols, fp);
	
	fclose(fp);

	for(m = 0; m < num_rows; m++){
		for(n = 0; n < num_cols; n++){
			fprintf(fr, "%3.2f ", matrix[m][n]);
		}	
		fprintf(fr,"\n");
	}
	fclose(fr);

}

void allocate_matrix(double ***matrix, int num_rows, int num_cols) {
	int i, j;

	*matrix = (double**)malloc(num_rows*sizeof(double*));
	(*matrix)[0] = (double*)malloc(num_rows*num_cols*sizeof(double));
	for (i = 1; i < num_rows; i++)
		(*matrix)[i] = (*matrix)[i-1] + num_cols;

	for(i = 0; i < num_rows; i++) {
		for (j = 0; j < num_cols; j++) {
			(*matrix)[i][j] = 0.;
		}
	}

}

void MatrixMultiply(int temp_index, int num_rows_A, int num_cols_A, double **matrix_A,  
						  int num_rows_B, int num_cols_B, double **matrix_B, double **matrix_C) {
	int i, j, k;

	/* The arrays A, B and C are shared among the threads along with their dimensions. 
		The "iterators" are private to each thread.
		The scheduling class is made static, so the iteration space is split into equal chunks. */
	#pragma omp parallel for private(i, j, k) schedule(static) \
		shared(matrix_A, matrix_B, matrix_C, num_rows_A, num_cols_A, num_rows_B, num_cols_B)
	for (i = 0; i < num_rows_A; i++) { 
		for (j = 0; j < num_cols_B; j++) {
			matrix_C[i][j] = 0.;
			for (k = temp_index; k < (num_rows_B + temp_index); k++) {
				matrix_C[i][j]+= matrix_A[i][k]*matrix_B[k-temp_index][j];
			}
		}
	}
}

int main(int argc, char *argv[]) {
	double **matrix_A=NULL, **matrix_B=NULL, **matrix_C=NULL, **my_matrix_A=NULL, **my_matrix_B=NULL, **my_matrix_C=NULL;
	int num_rows_A, num_cols_A, num_rows_B, num_cols_B;
	int my_num_rows_A, my_num_cols_A, my_num_rows_B, my_num_cols_B, max_num_rows_B;
	char *input_filename1, *input_filename2, *output_filename;
	int my_rank, numprocs, d_rank, u_rank, i, tracker, iters;
	int *displs_A, *sendcounts_A, *displs_B, *sendcounts_B, *displs_C, *sendcounts_C;

	if (argc < 3) {
		fprintf(stderr, "usage: %s bin-file\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	/* Read from command line: input_filename1, input_filename2, output_filename */
	input_filename1 = argv[1];
	input_filename2 = argv[2];
	output_filename = argv[3];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* Let process 0 read the A and B matrices from the two data files */
	if (my_rank == 0) {
		read_matrix_binaryformat(input_filename1, &matrix_A, &num_rows_A, &num_cols_A);
		read_matrix_binaryformat(input_filename2, &matrix_B, &num_rows_B, &num_cols_B);
		allocate_matrix(&matrix_C, num_rows_A, num_cols_B);
	}

	/* Broadcast the dimensions of A and B from process with rank 0 to all the other processes in the communicator */
	MPI_Bcast(&num_rows_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_cols_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_rows_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_cols_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Let process 0 distribute the pieces of A and B, by 1D partitioning, to all the other processes */
	displs_A = (int*)malloc(numprocs*sizeof(int));
	sendcounts_A = (int*)malloc(numprocs*sizeof(int));
	displs_B = (int*)malloc(numprocs*sizeof(int));
	sendcounts_B = (int*)malloc(numprocs*sizeof(int));
	displs_C = (int*)malloc(numprocs*sizeof(int));
	sendcounts_C = (int*)malloc(numprocs*sizeof(int));

	for(i = 0; i < numprocs; i++) {
		displs_A[i] = BLOCK_LOW(i, numprocs, num_rows_A)*num_cols_A;
		sendcounts_A[i]  = BLOCK_SIZE(i, numprocs, num_rows_A)*num_cols_A;
		displs_B[i] = BLOCK_LOW(i, numprocs, num_rows_B)*num_cols_B;
		sendcounts_B[i] = BLOCK_SIZE(i, numprocs, num_rows_B)*num_cols_B;
		displs_C[i] = displs_A[i];
		sendcounts_C[i] = BLOCK_SIZE(i, numprocs, num_rows_A)*num_cols_B;
	}
	/* since sendcounts[i] contains the density of each matrix block, dividing by 
		number of columns in each block yields number of rows in each block */
	my_num_rows_A = sendcounts_A[my_rank]/num_cols_A;
	my_num_cols_A = num_cols_A; // we got 1D partitioning 
	my_num_rows_B = sendcounts_B[my_rank]/num_cols_B;
	my_num_cols_B = num_cols_B;

	printf("I am rank %d, and number of rows is %d\n", my_rank, my_num_rows_B);


	/* Finding the block matrix in matrix B with largest number of rows, and use this value 
		as row-dimension for the matrix blocks of B. This is due to row-wise partitioning of matrix B,
		to avoid out of bounds-error when multiplying. This value is distributed to all the processes. */
	MPI_Allreduce(&my_num_rows_B, &max_num_rows_B, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	
	allocate_matrix(&my_matrix_A, my_num_rows_A, my_num_cols_A);
	allocate_matrix(&my_matrix_B, max_num_rows_B, my_num_cols_B);
	allocate_matrix(&my_matrix_C, my_num_rows_A, my_num_cols_B);

	/* Scatter matrix A and B to block matrices to all the other processes */
	//MPI_Scatter(*matrix_A, sendcounts_A[my_rank], MPI_DOUBLE, *my_matrix_A, sendcounts_A[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//MPI_Scatterv(matrix_A[0], sendcounts_A, displs_A, MPI_DOUBLE, *my_matrix_A[0], sendcounts_A[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Scatterv(matrix_B[0], sendcounts_B, displs_B, MPI_DOUBLE, my_matrix_B[0], sendcounts_B[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* Finding neighbouring ranks in the communicator with respect to my_rank, i.e. the processors with local matrix blocks under and over my_rank */
	d_rank = my_rank ? (my_rank - 1) : (numprocs - 1);
	u_rank = my_rank != (numprocs - 1) ? (my_rank + 1) : 0;
	
	/* Each process stores the vertical position in matrix B (and horisontal position in matrix A when multiplying), 
		so the multiplication is obtained by adding up partial results of a block in A and its corresponding block in B */
	tracker = displs_B[my_rank]/num_cols_B;

	/**
	iters = 0;
	while (iters < numprocs) {
		 Calculate C = A*B in parallel
		MatrixMultiply(tracker, my_num_rows_A, my_num_cols_A, my_matrix_A, max_num_rows_B, my_num_cols_B, my_matrix_B, my_matrix_C);
		
		 Send the matrix blocks in B upwards (with wraparound) along with the tracker for the next partial multiplication 
		MPI_Sendrecv_replace(my_matrix_B[0], max_num_rows_B*my_num_cols_B, MPI_DOUBLE, d_rank, 0, u_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(&tracker, 1, MPI_INT, d_rank, 1, u_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		iters++;
	}
*/
	/* Let process 0 gather, from all the other processes, the different pieces of C */
	//MPI_Gatherv(my_matrix_C[0], sendcounts_C[my_rank], MPI_DOUBLE, matrix_C[0], sendcounts_C, displs_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	

	/* Let process 0 write out the entire C matrix to an output data file */
	if (my_rank == 0){
		write_matrix_binaryformat(output_filename, matrix_C, num_rows_A, num_cols_B);
		
		/* Deallocating arrays stored for process 0 */
		free(matrix_A[0]); free(matrix_A);
		free(matrix_B[0]); free(matrix_B);
		free(matrix_C[0]); free(matrix_C);
	}

	/* Various array deallocation */
	free(displs_A); free(sendcounts_A);
	free(displs_B); free(sendcounts_B);
	free(displs_C); free(sendcounts_C);

	MPI_Finalize();
	return 0;
}