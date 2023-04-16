#include "test.h"
#include "log.h"

#include <iostream>
#include <stdexcept>
#include <mpi.h>
#include <chrono>

#include "torch_pch.h"
#include "agent.h"
#include "env.h"
#include "messages.h"
#include "model.h"

#include <iomanip>

using namespace std;

int test(int rank, int size, Args args)
{
    // Testing environment
    AtariEnv env(args.env_name, args.display_test);

    // Initialize the model.
    LSTMModel model(
        env.get_screen_channels(),
        env.get_num_actions()
    );

    // Initialize the agent.
    Agent agent(model, env, args);

    // Print a message indicating the testing loop started.
    log_info("Started testing process");

    // Set an initial time stamp.
    auto start_time = chrono::high_resolution_clock::now();

    float reward_total_sum = 0, reward_sum = 0; // total reward and reward for the current episode
    int num_tests = 0;

    while (1)
    {
        // Request the latest model parameters from the scheduler.
        sendInt(0, rank);
        sendInt(0, MSG_GET_GLOBAL_MODEL);

        // Expect the next received message to be the latest model parameters (or a stop message).
        int recv_type = recvInt(0);

        if (recv_type == MSG_STOP)
            break;

        if (recv_type != MSG_GLOBAL_MODEL)
            throw runtime_error("unexpected message type");

        // Receive serialized model parameters from the scheduler.
        std::vector<char> parameter_buf = recvBuffer(0);
        agent.model.deserialize(parameter_buf);
        agent.model.eval();

        if (args.gpu_id >= 0)
        {
            model.to(torch::kCUDA);
        }

        // Receive federation status
        int F_time = recvInt(0);
        int update_count = recvInt(0);
        int trajectories = recvInt(0);

        for (int step = 0; step < args.test_steps; ++step)
        {
          agent.action_test();
          reward_sum += agent.reward;

          if (agent.done)
          {
            // Print the elapsed CPU time, episode length, total reward and mean reward.
            auto end_time = chrono::high_resolution_clock::now();
            auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

            num_tests++;
            reward_total_sum += reward_sum;
            float mean_reward = reward_total_sum / num_tests;

            log_info("F_time %d | eps len %d | reward %f | mean reward %f", F_time, agent.eps_len, reward_sum, mean_reward);

            // Print the elapsed CPU time in HH:MM:SS format, episode length, total reward and mean reward.
            /*cout << "Elapsed time: " << setw(2) << elapsed_time / 1000 / 60 / 60 << ":" << setw(2) << elapsed_time / 1000 / 60 % 60 << ":" << setw(2) << elapsed_time / 1000 % 60 << " | ";
            cout << "Episode length: " << agent.eps_len << " | ";
            cout << "Total reward: " << reward_sum << " | ";
            cout << "Mean reward: " << mean_reward << endl;*/

            // Reset the environment.
            agent.env.reset();
            agent.clear_actions();
            agent.done = false;

            agent.eps_len = 0;
            reward_sum = 0;

            // Sleep for 10 seconds.
            this_thread::sleep_for(chrono::seconds(10));

            break;
          }
        }
    }

    return 0;
}
