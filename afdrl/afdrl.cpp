#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include "log.h"
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

  if (args.debug)
    log_set_debug();

  // Load the Atari environment config
  EnvConfig config;
  std::string rom_path = args.roms;

  if (args.env_name == "pong")
  {
    rom_path = rom_path + "pong.bin";
    config.crop_x = 0;
    config.crop_y = 34;
    config.crop_width = 160;
    config.crop_height = 160;
    config.frame_skip = 4;
    config.frame_stack = 3;
    config.max_episode_length = 10000;
  } else {
    if (rank == 0)
      std::cerr << "Unknown environment: " << args.env_name << std::endl;

    return -1;
  }

  // If we are the master process, start the scheduler loop.
  if (rank == 0)
  {
      // Start the scheduler loop.
      retcode = schedule(rank, size, args, rom_path, config);
  }

  // If we are the secondary process, start the testing loop.
  else if (rank == 1)
  {
      // Start the testing loop.
      retcode = test(rank, size, args, rom_path, config);
  }

  // Otherwise, start parallel training loops.
  else {
    retcode = train(rank, size, args, rom_path, config);
  }

  // Finalize the MPI environment.
  if (MPI_Finalize())
    throw runtime_error("mpi finalize fail");
}
