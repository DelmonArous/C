#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	int my_rank, size;
	int i;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for (i = my_rank; i < 65536; i += size) {
		/* chech_circuit(my_rank, i) */
	}

	printf("Process %d is done\n", my_rank);
	fflush(stdout);

	MPI_Finalize();
	return 0;
}