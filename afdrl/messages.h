#pragma once

#include <mpi.h>

#include <vector>
#include <stdexcept>

static const int MSG_GET_GLOBAL_MODEL = 0;
static const int MSG_UPDATE_GLOBAL_MODEL = 4;
static const int MSG_GLOBAL_MODEL = 0;
static const int MSG_GET_SCHEDULE = 2;
static const int MSG_SCHEDULE = 3;
static const int MSG_STOP = 5;

/**
 * Receive an integer from an MPI process.
 *
 * @param source The MPI process to receive from.
 * @return The integer received.
 */
static int recvInt(int source)
{
    int value;
    if (MPI_Recv(&value, 1, MPI_INT, source, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
      throw std::runtime_error("MPI_Recv failed");
    return value;
}

/**
 * Receive a byte array from an MPI process.
 * The first integer received is the length of the array.
 *
 * @param source The MPI process to receive from.
 * @return The byte array received.
 */
static std::vector<char> recvBuffer(int source)
{
    int length = recvInt(source);
    std::vector<char> bytes(length);
    if (MPI_Recv(bytes.data(), length, MPI_BYTE, source, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
      throw std::runtime_error("MPI_Recv failed");
    return bytes;
}

/**
 * Send an integer to an MPI process.
 *
 * @param dest The MPI process to send to.
 * @param value The integer to send.
 */
static void sendInt(int dest, int value)
{
    if (MPI_Send(&value, 1, MPI_INT, dest, 0, MPI_COMM_WORLD))
      throw std::runtime_error("MPI_Send failed");
}

/**
 * Send a byte array to an MPI process.
 * The first integer sent is the length of the array.
 *
 * @param dest The MPI process to send to.
 * @param bytes The byte array to send.
 */
static void sendBuffer(int dest, const std::vector<char>& bytes)
{
    sendInt(dest, bytes.size());
    if (MPI_Send(bytes.data(), bytes.size(), MPI_BYTE, dest, 0, MPI_COMM_WORLD))
      throw std::runtime_error("MPI_Send failed");
}
