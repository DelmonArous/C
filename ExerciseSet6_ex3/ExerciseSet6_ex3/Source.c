#include "mpi.h"
#include <malloc.h>
#include <math.h>

#define M_PI 3.14

int main(int argc, char* argv[]) {
	int size, my_rank, i;
	int my_start, my_stop;
	int M = 100;

	/* allocating three 1D arrays um, u, up of length M+2 */
	double *um = (double*)malloc((M+2)*sizeof(double));
	double *u = (double*)malloc((M+2)*sizeof(double));
	double *up = (double*)malloc((M+2)*sizeof(double));
	
	double x, dx = 1.0/(M+1);
	double t, dt = dx;
	double *tmp;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	for (i = my_rank; i <= M+1; i+=size) {
		x = i*dx;
		um[i] = sin(2.0*M_PI*x);
	}
	
	/*1 -> M*/

	my_start = (M*my_rank/size) + 1;
	my_stop = M*(my_rank+1)/size;

	for (i = my_start; i <= my_stop; i++) {
		u[i] = um[i] + 0.5*(um[i-1]-2*um[i]+um[i+1]);
	}

	u[0] = u[M+1] = 0.0;
	
	t = dt;
	while (t < 1.0) {
		t += dt;
		for (i = my_start; i <= my_stop; i++) {
			up[i] = um[i]+u[i-1]+u[i+1];
		}
		up[0] = up[M+1];
		
		/* shuffle the three arrays */
		tmp = um;
		um = u;
		u = up;
		up = tmp;
	}

	return 0;
}