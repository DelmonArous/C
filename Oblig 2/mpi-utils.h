#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

/* rank of root process */
#define ROOT_PROCESS 0

/* check if rank is root process */
#define PROCESS_IS_ROOT(rank)			\
  ((rank) == ROOT_PROCESS)

/* lowest index in block distribution */
#define BLOCK_LOW(id, p, n)			\
  ((id)*(n)/(p))

/* highest (non-inclusive) index in block distribution */
#define BLOCK_HIGH(id, p, n)			\
  BLOCK_LOW((id)+1, (p), (n))

/* number of indices assigned in block distribution */
#define BLOCK_SIZE(id, p, n)			\
  (BLOCK_HIGH((id), (p), (n)) - BLOCK_LOW((id), (p), (n)))

/* finalizes and exits with status */
#define FINALIZE_AND_EXIT(status)		\
  do {						\
    MPI_Finalize();				\
    exit((status));				\
  } while (0)

/* finalizes and exits signaling success */
#define FINALIZE_AND_SUCCEED			\
  FINALIZE_AND_EXIT(EXIT_SUCCESS)

/* finalizes and exits signaling failure */
#define FINALIZE_AND_FAIL			\
  FINALIZE_AND_EXIT(EXIT_FAILURE);

/* does a printf from root process only */
#define root_printf(id, ...)			\
  (PROCESS_IS_ROOT(id) ? printf(__VA_ARGS__) : 0)

/* does an fprintf to stderr from root process only */
#define root_eprintf(id, ...)			\
  (PROCESS_IS_ROOT(id) ? fprintf(stderr, __VA_ARGS__) : 0)

#endif /* MPI_UTILS_H */
