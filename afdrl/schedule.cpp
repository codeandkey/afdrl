/**
 * @file schedule.cpp
 * @brief AFDRL scheduler
 */

#include "schedule.h"
#include "log.h"

#include <iostream>
#include <stdexcept>
#include <random>
#include <set>
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
 * Client updates are processed in strictly increasing order of client index. TODO enforce
*/

class ClientSchedule {
  public:
    ClientSchedule(int seed, int minspace, int maxspace, int minlen, int maxlen, int steps_ratio, int steps_var, int chan, int actions)
      : length_dist(minlen, maxlen),
        space_dist(minspace, maxspace),
        steps_ratio(steps_ratio),
        steps_var(steps_var),
        model(chan, actions)
    {
      // Initialize job rng
      job_rng.seed(seed);

      start_time = -1;
      end_time = -1;

      advance(0);
    }

    enum Status {
      PENDING, // the job has not started yet
      WAITING, // the job has started, but no response received yet
      EARLY,   // the job completed before the finish time step
    } status;

    int start_time;
    int end_time;
    int steps;
    int steps_var, steps_ratio;

    LSTMModel model;

    /**
     * Advance the schedule sequence.
     * @param t The current time step
     */
    void advance(int t)
    {
      if (t < end_time)
        throw runtime_error("Cannot advance schedule before end time");

      // Generate new job
      start_time = t + 1 + space_dist(job_rng);
      end_time = start_time + length_dist(job_rng);

      // Generate new number of steps
      int expected_steps = steps_ratio * (end_time - start_time);
      normal_distribution<float> step_dist(expected_steps, steps_var);

      steps = round(step_dist(job_rng));

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

void merge_model(LSTMModel& dest, ClientSchedule& from)
{
  // TODO: merge strategy
  dest.add(from.model, 1.0f);

  // Write the model update to stdout
  cout << "===> Global model updates from " << from.start_time << " -> " << from.end_time << " over " << from.steps << " steps" << endl;
  cout << "Delta:" << endl;
  from.model.print();
  cout << "====\nNew parameters:" << endl;
  dest.print();
  cout << "<===" << endl;
}

int schedule(int rank, int size, Args args)
{
  mpi_size = size;

  // Initialize a shared global environment (for parameters)
  AtariEnv* env = new AtariEnv(args.env_name, false);

  // Initialize the shared global model
  LSTMModel model(
      env->get_screen_channels(),
      env->get_num_actions()
  );

  // Set CTRL-C handler
  signal(SIGINT, sigint_handler);

  // Initialize counters
  int total_updates = 0;
  int total_trajectories = 0;

  // Initialize client schedule streams
  vector<ClientSchedule> schedules;

  for (int i = 0; i < args.num_clients; ++i)
  {
    schedules.emplace_back(
      i,
      0, 0, // no spacing for now
      args.min_offline_time,
      args.max_offline_time,
      args.steps_ratio,
      args.steps_var,
      env->get_screen_channels(),
      env->get_num_actions()
    );
  }

  delete env;
  env = nullptr;

  int F_time = 0;

  while (F_time < args.num_steps)
  {
    // We will process all updates required at this timestep
    // First, collect any schedules pending to start now.

    set<int> pending;

    for (int i = 0; i < schedules.size(); i++)
    {
      if (schedules[i].status == ClientSchedule::PENDING && schedules[i].start_time == F_time)
      {
        // Found a pending job
        pending.insert(i);
      }
    }

    // We then collect jobs waiting to join at this timestep
    set<int> waiting;

    for (int i = 0; i < schedules.size(); i++)
    {
      if (schedules[i].end_time != F_time)
        continue;

      // The job will join on this timestep. Is it already complete?
      if (schedules[i].status == ClientSchedule::EARLY)
      {
        // The job is already complete
        // Merge the waiting parameters and advance the job

        merge_model(model, schedules[i]);
        schedules[i].advance(F_time);
      }
      else
      {
        // The job is not complete
        // We must wait for the job to complete
        waiting.insert(i);
      }
    }

    // We have lists of active jobs. Continue processing messages until each
    // required job is complete.

    while (!waiting.empty() || !pending.empty())
    {
      // Read next message source
      int source = recvInt(MPI_ANY_SOURCE);

      // Read next message
      int msg = recvInt(source);

      // Handle message
      switch (msg)
      {
        case MSG_GET_SCHEDULE:
          {
            // If there is a pending job, send it over.
            // Otherwise, tell the client to sleep.

            if (pending.empty())
            {
              sendInt(source, MSG_SLEEP);
              break;
            }

            int i = *pending.begin();
            
            // Send schedule information
            sendInt(source, MSG_SCHEDULE);            // Message type
            sendInt(source, schedules[i].steps);      // Number of steps
            sendInt(source, i);                       // Schedule index

            vector<char> params = model.serialize();
            sendBuffer(source, params); // Model parameters

            // Check sanity
            if (schedules[i].status != ClientSchedule::PENDING)
              throw runtime_error("Invalid schedule status");
            if (schedules[i].end_time <= F_time)
              throw runtime_error("Invalid schedule end time");

            // Write debug info
            log_debug("Sent schedule %d to %d", i, source);

            // Mark job as waiting
            schedules[i].status = ClientSchedule::WAITING;
            pending.erase(i);
          }
          break;
        case MSG_UPDATE_GLOBAL_MODEL:
          // The client has an update for us
          {
            // Receive job index
            int i = recvInt(source);

            // Receive update parameters
            vector<char> buffer = recvBuffer(source);
            schedules[i].model.deserialize(buffer);

            // Sanity check
            if (schedules[i].status != ClientSchedule::WAITING)
              throw runtime_error("Invalid schedule status");

            // If the model is joining later, we wait for later timesteps
            if (schedules[i].end_time > F_time)
            {
              // Mark job as early
              schedules[i].status = ClientSchedule::EARLY;
              break;
            }

            // Otherwise, the job is merging now
            merge_model(model, schedules[i]);
            schedules[i].advance(F_time);

            // remove from waiting list
            waiting.erase(i);
          }
          break;
        case MSG_GET_GLOBAL_MODEL:
          // Send global model
          {
            // Send message type
            sendInt(source, MSG_GLOBAL_MODEL);
            
            // Serialize global model to buffer
            vector<char> buffer = model.serialize();
            sendBuffer(source, buffer);

            // Send federation time
            sendInt(source, F_time);

            // Send global update count
            sendInt(source, total_updates);

            // Send total trajectory count
            sendInt(source, total_trajectories);
          }
          break;
        default:
          // Unknown message
          throw runtime_error("Unknown message");
      }
    }

    cout << "finished F_time = " << F_time << endl;
    F_time += 1;
  }

  return 0;
}
