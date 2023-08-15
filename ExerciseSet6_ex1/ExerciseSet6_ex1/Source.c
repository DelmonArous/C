#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	int i, N = 4;
	int my_rank, size, my_rank_partner;
	int message = 0;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//printf("Hello world, I've rank %d out of %d procs.\n", my_rank,size);

	if (my_rank % 2) { // my_rank is odd
		my_rank_partner = my_rank - 1;
	}
	else { // my_rank is even
		my_rank_partner = my_rank + 1;
	}

	for (i = 0; i < N; i++) {
		if (!(my_rank % 2)) {
			MPI_Send(&message, 1, MPI_INT, my_rank_partner, 999, MPI_COMM_WORLD);
			MPI_Recv(&message, 1, MPI_INT, my_rank_partner, 999, MPI_COMM_WORLD, &status);
		}
		else {
			MPI_Recv(&message, 1, MPI_INT, my_rank_partner, 999, MPI_COMM_WORLD, &status);
			MPI_Send(&message, 1, MPI_INT, my_rank_partner, 999, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}