#include <omp.h>

int main(int argc, char *argv[]) {
	double sum = 0, add = 10;

	#pragma omp parallel shared(sum) private(add)
	{
		#pragma omp critical
		sum = sum + add;
	}

	return 0;
}