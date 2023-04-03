/**
 * @file schedule.h
 * @brief AFDRL scheduler
 */

#ifndef AFDRL_SCHEDULE_H
#define AFDRL_SCHEDULE_H

#include "args.h"

/**
 * Starts the scheduler loop.
 * 
 * @param rank The rank of the scheduler.
 * @param size The size of the MPI communicator.
 * @param args The configuration arguments.
 * @return int The exit code.
 */
int schedule(int rank, int size, Args args);

#endif // AFDRL_SCHEDULE_H
