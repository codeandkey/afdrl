/**
 * @file train.h
 * @brief Training loop for AFDRL
 */

#ifndef AFDRL_TRAIN_H
#define AFDRL_TRAIN_H

#include "args.h"
#include "env.h"

/**
 * Starts a training client.
 * 
 * @param rank The rank of the client.
 * @param size The size of the MPI communicator. 
 * @param args The configuration arguments.
 * @return int The exit code.
 */
int train(int rank, int size, Args args, std::string rom_path, EnvConfig config);

#endif
