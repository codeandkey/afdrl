#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include "args.h"
#include "model.h"
#include "train.h"
#include "test.h"
#include "schedule.h"

using namespace std;

int main(int argc, char** argv)
{
  int size, rank, retcode;

  // Initialize the MPI environment
  if (MPI_Init(&argc, &argv))
    throw runtime_error("comm size query fail");

  // Check the number of processes
  if (MPI_Comm_size(MPI_COMM_WORLD, &size))
    throw runtime_error("comm size query fail");

  // Check the rank of the process
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank))
    throw runtime_error("comm rank query fail");

  // Parse command line arguments
  Args args(argc, argv);

  if (args.help)
  {
    // Show usage information on the master process.
    if (rank == 0)
      args.usage(argv);

    // Finalize the MPI environment.
    if (MPI_Finalize())
      throw runtime_error("mpi finalize fail");

    return 0;
  }

  // If we are the master process, start the scheduler loop.
  if (rank == 0)
  {
      // Start the scheduler loop.
      retcode = schedule(rank, size, args);
  }

  // If we are the secondary process, start the testing loop.
  else if (rank == 1)
  {
      // Start the testing loop.
      retcode = test(rank, size, args);
  }

  // Otherwise, start parallel training loops.
  else {
    retcode = train(rank, size, args);
  }

  // Finalize the MPI environment.
  if (MPI_Finalize())
    throw runtime_error("mpi finalize fail");
}
