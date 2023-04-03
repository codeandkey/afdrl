/**
 * @file schedule.cpp
 * @brief AFDRL scheduler
 */

#include "schedule.h"

#include <iostream>
#include <stdexcept>
#include <random>
#include <mpi.h>

#include <signal.h>

#include "messages.h"
#include "model.h"

using namespace std;

/**
 * The scheduler simulates a continuous stream of discrete time steps, with a
 * collection of 
 */

/**
 * The scheduler process decides a strict order in which clients will update,
 * and how long each client will be offline for. At every time step, the
 * scheduler decides which clients will rendevous with the federation.
 *
 * Workers receive a global model, the client's associated environment, and
 * the number of steps to train the model for.
 *
 * The scheduler receives updates from the workers, which are stored in a
 * pending updates queue.
 *
 * It is critical that jobs for the same client index do not overlap.
 *
 * Client updates are processed in strictly increasing order of client index.
*/

/**
 * Describes the status of a single client in the federation.
 * A client job is pending if it is waiting to be scheduled.
 * A client job is waiting if it is currently being trained.
 * 
 */
struct ClientStatus {
  int start;  // start time of the current job
  int end;    // end time of the current job
  int steps;  // number of offline steps
  
  enum Status {
    PENDING,
    WAITING,
  } status;

  ClientStatus(int start, int end, int steps, Status status)
    : start(start), end(end), steps(steps), status(status) {}
};

/**
 * Generates a seeded random sequence of ClientStatus jobs
 */
class ClientStatusGenerator {
  public:
    ClientStatusGenerator(int seed);

  private:
    random_device length_rng;
    random_device space_rng;
};

class ClientSchedule {
  public:
    ClientSchedule(int seed, int minspace, int maxspace, int minlen, int maxlen, int minsteps, int maxsteps)
      : length_dist(minlen, maxlen),
        space_dist(minspace, maxspace),
        step_dist(minsteps, maxsteps)
    {
      // Initialize job rng
      job_rng.seed(seed);

      advance(0);
    }

    enum Status {
      PENDING,
      WAITING,
    } status;

    int start_time;
    int end_time;
    int steps;

    /**
     * Advance the schedule sequence.
     * @param t The current time step
     */
    void advance(int t)
    {
      if (t < end_time)
        throw runtime_error("Cannot advance schedule before end time");

      // Generate new job
      start_time = t + space_dist(job_rng);
      end_time = start_time + length_dist(job_rng);

      // Generate new number of steps
      steps = step_dist(job_rng);

      // Set status to pending
      status = PENDING;
    }

  private:
    mt19937 job_rng;

    uniform_int_distribution<int> length_dist, space_dist, step_dist;
};

static int mpi_size;

void sigint_handler(int sig)
{
  // Send STOP to all test/train processes
  for (int i = 1; i < mpi_size; i++)
  {
    sendInt(i, MSG_STOP);
  }
}

int schedule(int rank, int size, Args args)
{
    mpi_size = size;

    // Initialize a shared global environment (for parameters)
    AtariEnv env(args.env_name, false);

    // Initialize the shared global model
    LSTMModel model(
        env.get_screen_channels(),
        env.get_num_actions()
    );

    // Initialize a model for each simulated client
    vector<LSTMModel> clients;
    for (int i = 0; i < args.num_clients; i++)
    {
      clients.push_back(LSTMModel(
        env.get_screen_channels(),
        env.get_num_actions()
      ));
    }

    // Set CTRL-C handler
    signal(SIGINT, sigint_handler);

    // Initialize counters
    int total_updates = 0;
    int total_trajectories = 0;

    // Initialize client schedule streams
    vector<ClientSchedule> schedules;

    int current_time_step = 0;

    /**
     * The server only advances the timestep once no jobs have a start or end time
     * at the current timestep.
     */
    while (1)
    {
      // While there is work to do at this timestep, we receive updates and requests from clients.
      bool finished = true;

      // Read next message source
      int source = recvInt(MPI_ANY_SOURCE);

      // Read next message
      int msg = recvInt(source);

      // Handle message
      switch (msg)
      {
        case MSG_GET_SCHEDULE:
          // Check if any of the schedule streams have a pending job ready
          {
            // Check if any of the schedules have a pending job ready
            bool found = false;
            for (int i = 0; i < schedules.size(); i++)
            {
              if (schedules[i].status == ClientSchedule::PENDING)
              {
                // Found a pending job
                found = true;

                // Send message type
                sendInt(source, MSG_SCHEDULE);

                // Send client index
                sendInt(source, i);

                // Send start time
                sendInt(source, schedules[i].start_time);

                // Send end time
                sendInt(source, schedules[i].end_time);

                // Send number of steps
                sendInt(source, schedules[i].steps);

                // Write debug info
                cout << "Sent schedule to " << source << endl;

                // Mark job as waiting
                schedules[i].status = ClientSchedule::WAITING;

                // Break out of loop
                break;
              }
            }

            // If no pending jobs were found, send a sleep message back to the client
            if (!found)
            {
              // Send message type
              sendInt(source, MSG_SLEEP);

              // Write debug info
              cout << "Sent sleep message to " << source << endl;
            }
          }
          throw runtime_error("Not implemented");
          break;
        case MSG_UPDATE_GLOBAL_MODEL:
          // Update global model
          // TODO: Implement
          throw runtime_error("Not implemented");
          break;
        case MSG_GET_GLOBAL_MODEL:
          // Send global model
          {
            // Send message type
            sendInt(source, MSG_GLOBAL_MODEL);
            
            // Serialize global model to buffer
            vector<char> buffer = model.serialize();
            sendBuffer(source, buffer);

            // Send global update count
            sendInt(source, total_updates);

            // Send total trajectory count
            sendInt(source, total_trajectories);

            // Write debug info
            cout << "Sent global model to " << source << endl;
          }
          break;
        default:
          // Unknown message
          throw runtime_error("Unknown message");
      }
    }

    return 0;
}
