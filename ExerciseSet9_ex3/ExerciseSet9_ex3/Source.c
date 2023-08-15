#include "omp.h"
#include <stdio.h>
#include <stdlib.h>

void vectorinnerproduct(double *a, double *b, double *c, int N, int chunk) {
	int i;
	#pragma omp parallel for shared(a,b,c,N,chunk) private(i) schedule(static,chunk) num_threads(4) 
	{
		for(i = 0; i < N; i++) {
			c[i] = a[i] + b[i];
		}
	}
}

int main() {
	int i, N = 1000, chunk = 100;
	double startTime, stopTime;
	double *a, *b, *c;
	a = (double*)malloc(N*sizeof(double));
	b = (double*)malloc(N*sizeof(double));
	c = (double*)malloc(N*sizeof(double));

	for (i = 0; i < N; i++) {
		a[i] = b[i] = i*1.0;
	}

	startTime = omp_get_wtime();
	vectorinnerproduct(a, b, c, N, chunk);
	stopTime = omp_get_wtime();

	printf("Time usage: %f s", stopTime - startTime);

	free(a); free(b); free(c);

	return 0;
}