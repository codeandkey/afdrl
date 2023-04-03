/**
 * @file test.h
 * @brief Testing loop for AFDRL
 */

#ifndef AFDRL_test_H
#define AFDRL_test_H

#include "model.h"
#include "messages.h"
#include "args.h"

/**
 * Starts a testing client.
 * 
 * @param rank The rank of the client.
 * @param size The size of the MPI communicator. 
 * @param args The configuration arguments.
 * @return int The exit code.
 */
int test(int rank, int size, Args args);

#endif
