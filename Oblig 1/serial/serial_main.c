#include <stdio.h>
#include <stdlib.h>
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

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters) {
	int i, j, counter;
	float** temp;

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

		counter++;
	}
	u_bar->m = u->m;
	u_bar->n = u->n;
	u_bar->image_data = u->image_data;
}

int main(int argc, char *argv[]) {
	int m, n, c, iters;
	float kappa;
	image u, u_bar;
	unsigned char *image_chars;
	char *input_jpeg_filename, *output_jpeg_filename;
	
	if (argc < 2) {
		fprintf(stderr, "usage: %s jpeg-file\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	/* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
	kappa = (float) atof(argv[1]);
	iters = atoi(argv[2]);
	input_jpeg_filename = argv[3];
	output_jpeg_filename = argv[4];
	import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);

	/* allocate the 2d array inside u and u_bar, when m and n are given as input */
	allocate_image(&u, m, n);
	allocate_image(&u_bar, m, n);
	/* convert a 1d array of type unsigned char values into an image struct */
	convert_jpeg_to_image(image_chars, &u);

	/* carries out iters iterations of the isotropic diffusion on a noisy image object u */
	/* the denoised image is stored in the u_bar object */
	iso_diffusion_denoising(&u, &u_bar, kappa, iters);

	convert_image_to_jpeg(&u_bar, image_chars);
	export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);

	/* free storage used by the 2d array image_data inside u and u_bar */
	deallocate_image(&u);
	deallocate_image(&u_bar);

	return 0;
}
