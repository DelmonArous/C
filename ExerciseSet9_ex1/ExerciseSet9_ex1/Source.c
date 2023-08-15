#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void ex_a(double *a, double *b) {
	int i;
	double x = 6.8;

	#pragma omp parallel for default(private) shared(a,b,x) private(i)
	{
		for (i = 0; i < (int) sqrt(x); i++) {
			a[i] = 2.3*x;
			if (i < 10) 
				b[i] = a[i];
		}
	}
}

void ex_b(double *a, double *b) {
	int i, n, flag = 0;

	/* Kan ikke paralleliseres? */
	#pragma omp parallel for default(private) shared(a,b,n,flag) private(i)
	{
		#pragma omp for nowait
		for (i = 0; (i < n) & (!flag); i++) {
			a[i] = 2.3*i;
			if (a[i] < b[i])
				flag = 1;
		}
	}
}

void ex_c(double *a) {
	int i, n;

	#pragma omp parallel for default(private) shared(a,n) private(i)
	{
		#pragma omp for nowait
		for (i = 0; i < n; i++)
			a[i] = foo(i);
	}
}

void ex_d(double *a, double *b) {
	int i, n;

	#pragma omp parallel for default(private) shared(a,b,n) private(i)
	{
		#pragma omp for nowait
		for (i = 0; i < n; i++) {
			a[i] = foo(i);
			if (a[i] < b[i]) 
				a[i] = b[i];
		}
	}
}

void ex_e(double *a, double *b) {
	int i, n;

	/* The number of loop iterations can not be non-deterministic; break not allowed inside the for-loop */
	for (i = 0; i < n; i++) {
		a[i] = foo(i);
		if (a[i] < b[i]) 
			break;
	}
}

void ex_f(double *a, double *b) {
	int i, n;
	double dotp = 0;

	#pragma omp parallel for default(private) shared(a,b,n) private(i) reduction(+:dotp)
	{
		#pragma omp for nowait
		for (i = 0; i < n; i++)
			dotp += a[i] * b[i];
	}
}

void ex_g(double *a) {
	int i, k;
	#pragma omp parallel for
	{
		for (i = k; i < 2*k; i++)
			a[i] = a[i] + a[i-k];
	}
}

void ex_h(double *a) {
	int i, k, n;
	double b;

	/* racing condition if n >= 2*k */
	for (i = k; i < n; i++)
		a[i] = b*a[i-k];
}

int main(int argc, char *argv[]) {
	return 0;
}