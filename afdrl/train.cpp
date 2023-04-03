#include "train.h"

#include <iostream>
#include <stdexcept>
#include <mpi.h>

#include "agent.h"
#include "messages.h"
#include "model.h"

using namespace std;

int train(int rank, int size, Args args)
{
    // Initialize local environment
    AtariEnv env(args.env_name, false);

    // Initialize the model.
    LSTMModel model(
        env.get_screen_channels(),
        env.get_num_actions()
    );

    // Initialize the optimizer.
    torch::optim::Adam optimizer(
        model.parameters(),
        torch::optim::AdamOptions(args.lr)
    );

    // Initialize the agent.
    Agent agent(model, env, args);

    // Print a message indicating the training loop started.
    cout << "Training loop started." << endl;

    while (1)
    {
        // Request a schedule from the scheduler.
        sendInt(0, rank);
        sendInt(0, MSG_GET_SCHEDULE);

        // Expect the next received message to be a schedule (or a stop message).
        int recv_type = recvInt(0);

        if (recv_type == MSG_STOP)
            break;

        if (recv_type != MSG_SCHEDULE)
            throw runtime_error("unexpected message type");
        
        // Receive the schedule length
        int schedule_length = recvInt(0);

        // Receive the client index
        int client_index = recvInt(0);

        // Receive the model parameters
        std::vector<char> parameter_buf = recvBuffer(0);
        agent.model.deserialize(parameter_buf);

        // Update the optimizer target parameters
        optimizer.param_groups()[0].params() = agent.model.parameters();

        // Receive the environment state
        std::vector<char> state_buf = recvBuffer(0);
        agent.env.deserialize(state_buf);

        // Run the scheduled work
        int total_steps = 0;
        while (total_steps < schedule_length)
        {
            // Reset the hidden and cell states if the environment is done.
            if (agent.done)
            {
                agent.hx = torch::zeros({1, 512}, torch::kFloat32);
                agent.cx = torch::zeros({1, 512}, torch::kFloat32);
            }

            // Move the model and the environment to the GPU if necessary.
            if (args.gpu_id >= 0)
            {
                agent.hx = agent.hx.to(torch::kCUDA);
                agent.cx = agent.cx.to(torch::kCUDA);
            }

            // Run the agent for a number of steps.
            for (int i = 0; i < args.num_steps; i++)
            {
                agent.action_train();
                total_steps += 1;

                if (agent.done)
                    break;
            }

            // Initialize the discounted return tensor.
            torch::Tensor R = torch::zeros({1, 1}, torch::kFloat32);

            if (!agent.done)
            {
                // Compute the discounted return.
                torch::IValue result = agent.model.forward(torch::TensorList({agent.state.unsqueeze(0), agent.hx, agent.cx}));
                R = result.toTensorList().get(0).detach();
            }

            // Move the discounted return tensor to the GPU if necessary.
            if (args.gpu_id >= 0)
                R = R.to(torch::kCUDA);

            agent.values.push_back(R);

            torch::Tensor policy_loss = torch::zeros({1}, torch::kFloat32);
            torch::Tensor value_loss = torch::zeros({1}, torch::kFloat32);
            torch::Tensor gae = torch::zeros({1}, torch::kFloat32);

            // Walk through the trajectory in reverse order.
            for (int i = agent.rewards.size() - 1; i >= 0; i--)
            {
                // Compute the discounted return.
                R = args.gamma * R + agent.rewards[i];

                // Compute the generalized advantage estimate.
                torch::Tensor delta = agent.rewards[i] + args.gamma * agent.values[i + 1] - agent.values[i];
                gae = gae * args.gamma * args.tau + delta;

                // Compute the policy loss.
                torch::Tensor log_prob = agent.log_probs[i];
                policy_loss = policy_loss - log_prob * gae.detach();
                policy_loss = policy_loss - 0.01f * agent.entropies[i];

                // Compute the value loss.
                torch::Tensor value = agent.values[i];
                value_loss = value_loss + torch::pow(R - value, 2);
            }

            // Zero the gradients.
            optimizer.zero_grad();

            // Backpropagate the loss.
            torch::Tensor loss = policy_loss + 0.5f * value_loss;
            loss.backward();

            // Clip the gradients.
            torch::nn::utils::clip_grad_norm_(agent.model.parameters(), 40.0f); // TODO: make this a parameter

            // Update the model parameters.
            optimizer.step();

            // Clear the trajectory.
            agent.clear_actions();
        }

        // Send the updated model parameters to the scheduler.
        sendInt(0, rank);
        sendInt(0, MSG_UPDATE_GLOBAL_MODEL);

        std::vector<char> parameter_buf = agent.model.serialize();
        sendBuffer(0, parameter_buf);

    }

    return 0;
}
