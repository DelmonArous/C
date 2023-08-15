#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include "mpi-utils.h"		
#include "../simple-jpeg/import_export_jpeg.h"

typedef struct {
	float** image_data; /* a 2D array of floats */
	int m; /* # pixels in x-direction */
	int n; /* # pixels in y-direction */
} image;

void allocate_image(image *u, int m, int n) {
	int i;

	u->m = m;
	u->n = n;

	u->image_data = (float**)malloc(m*sizeof(float*));
	for (i = 0; i < m; i++) {
		u->image_data[i] = (float*)malloc(n*sizeof(float));
	}
}

void deallocate_image(image *u) {
	int i;

	for (i = 0; i < u->m; i++) {
			free(u->image_data[i]);
	}
	free(u->image_data);
}

void convert_jpeg_to_image(const unsigned char* image_chars, image *u) {
	int i, j;

	/* the vaules in image_chars are stored in row-major order */
	for (i = 0; i < u->m; i++) {
		for (j = 0; j < u->n; j++) {
			u->image_data[i][j] = (float) image_chars[i*u->n + j];
		}
	}
}

void convert_image_to_jpeg(const image *u, unsigned char* image_chars) {
	int i, j;

	/* the vaules in image_chars are stored in row-major order */
	for (i = 0; i < u->m; i++) {
		for (j = 0; j < u->n; j++) {
			image_chars[i*u->n + j] = (unsigned char) u->image_data[i][j];
		}
	}
}

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters, int my_rank, int num_procs) {
	int i, j, counter;
	float** temp;
	int up_rank, down_rank;

	up_rank = my_rank != num_procs-1 ? my_rank + 1: MPI_PROC_NULL;
	down_rank = my_rank ? my_rank - 1 : MPI_PROC_NULL;

	// This includes the boundary conditions
	for(i = 0; i < u->m; i++){
		for(j = 0; j < u->n; j++){
			u_bar->image_data[i][j]= u->image_data[i][j];
		}
	}

	counter = 0;
	while (counter < iters) {
		for (i = 1; i < u->m - 1; i++) {
			for (j = 1; j < u->n - 1; j++) {
				/* the isotropic diffusion formula */
				u_bar->image_data[i][j] = u->image_data[i][j] + kappa*(u->image_data[i-1][j]+ u->image_data[i][j-1] - 
					4*u->image_data[i][j] + u->image_data[i][j+1] + u->image_data[i+1][j]);  
			}
		}
		/* let u_bar be the previous result for the next iteration */
		temp = u_bar->image_data;
		u_bar->image_data = u->image_data;
		u->image_data = temp;
		
		/* exchange overlapped regions */
		MPI_Sendrecv(u->image_data[1], u->n, MPI_FLOAT, down_rank, 0, 
			u->image_data[(u->m)-1], u->n, MPI_FLOAT, up_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Sendrecv(u->image_data[u->m-2], u->n, MPI_FLOAT, up_rank, 1, 
			u->image_data[0], u->n, MPI_FLOAT, down_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		counter++;
	}
	u_bar->m = u->m;
	u_bar->n = u->n;
}

int main(int argc, char *argv[]) {
	int m, n, c, iters, disp;
	int my_m, my_n, my_rank, num_procs;
	int *lower, *size;
	float kappa;
	image u, u_bar, whole_image;
	unsigned char *image_chars = NULL, *my_image_chars;
	char *input_jpeg_filename, *output_jpeg_filename;
	
	if (argc < 2) {
		fprintf(stderr, "usage: %s jpeg-file\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	/* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
	kappa = (float) atof(argv[1]);
	iters = atoi(argv[2]);
	input_jpeg_filename = argv[3];
	output_jpeg_filename = argv[4];
	import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);

	/* process with rank 0 imports jpeg-file into image_chars of length m*n as 1d */
	if (my_rank == 0) {
		import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
		// allocate_image(&whole_image, m, n);
	}

	/* boradcast the dimensions of the image from process with rank 0 to all the other processes in the communicator */
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* divide the m x n pixels evenly among the MPI processes, num_procs */
	/* lower[i] and size[i] contains the values of the lower index and image block size, respectively, for each processor i */
	lower = (int*)malloc(num_procs*sizeof(int));
	size  = (int*)malloc(num_procs*sizeof(int));

	for(disp = 0; disp < num_procs; disp++) {
		lower[disp] = (BLOCK_LOW(disp, num_procs, m-2))*n;
		size[disp]  = (BLOCK_SIZE(disp, num_procs, m-2)+2)*n;
	}
	my_m = size[my_rank]/n;
	my_n = n;

	allocate_image(&u, my_m, my_n);
	allocate_image(&u_bar, my_m, my_n);

	/* each process asks process 0 for a partitioned region of image_chars and copy the values into my_image_char of dimension my_m*my_n */
	my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));
	MPI_Scatterv(image_chars, size, lower, MPI_UNSIGNED_CHAR, 
		my_image_chars, size[my_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	/* each processor has now its own my_image_char of the whole image_chars, and converts it to u of type image */
	/* the isotropic diffusion starts and the smoothing-result gets stored in u */
	/* u of type image gets converted back into my_image_chars */
	convert_jpeg_to_image(my_image_chars, &u);
	iso_diffusion_denoising(&u, &u_bar, kappa, iters, my_rank, num_procs);
	convert_image_to_jpeg(&u, my_image_chars);

	/* each process sends its resulting content of u_bar to process 0 */
	/* process 0 recieves from each process incoming values and  */
	/* copy them into the designated region of struct whole_image */
	MPI_Gatherv(my_image_chars, size[my_rank], MPI_UNSIGNED_CHAR, 
		image_chars, size, lower, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	/* never really need whole_image since we gathered all the result from my_image_chars into image_chars */
	if (my_rank == 0) {
		//convert_image_to_jpeg(&whole_image, image_chars); 
		export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
		//deallocate_image(&whole_image);
	}

	deallocate_image(&u);
	deallocate_image(&u_bar);

	MPI_Finalize();
	return 0;
}
