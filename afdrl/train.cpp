#include "train.h"
#include "log.h"

#include <c10/core/TensorOptions.h>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

#include "agent.h"
#include "messages.h"
#include "model.h"

#include "torch_pch.h"

using namespace std;

int train(int rank, int size, Args args)
{
    // Initialize local environment
    AtariEnv env(args.env_name, true);

    LSTMModel model(
        env.get_screen_channels(),
        env.get_num_actions()
    );

    // Last client model, used to compute update difference
    LSTMModel init_model(
        env.get_screen_channels(),
        env.get_num_actions()
    );

    // Initialize the agent.
    Agent agent(model, env, args);

    // Print a message indicating the training loop started.
    log_debug("Started training process %d", rank);

    while (1)
    {
        // Request a schedule from the scheduler.
        sendInt(0, rank);
        sendInt(0, MSG_GET_SCHEDULE);

        // Expect the next received message to be a schedule (or a stop message).
        int recv_type = recvInt(0);

        if (recv_type == MSG_STOP)
            break;

        if (recv_type == MSG_SLEEP)
        {
            // Sleep for a little, and try again
            usleep(100000);
            continue;
        }

        if (recv_type != MSG_SCHEDULE)
            throw runtime_error("unexpected message type");
        
        // Receive the schedule length
        int schedule_length = recvInt(0);

        // Receive the client index
        int client_index = recvInt(0);

        // Receive the model parameters
        std::vector<char> parameter_buf = recvBuffer(0);
        agent.model.deserialize(parameter_buf);
        init_model.deserialize(parameter_buf);

        // Update the optimizer target parameters
        //optimizer.param_groups()[0].params() = agent.model.parameters();
        agent.model.train();

        // TODO: ideal if we can share this, but replacing param groups seems broken
        // Initialize the optimizer.
        torch::optim::Adam optimizer(
            agent.model.parameters(),
            torch::optim::AdamOptions(args.lr)
        );

        if (args.gpu_id >= 0)
        {
            init_model.to(torch::kCUDA);
            agent.model.to(torch::kCUDA);
        }

        log_debug("%d starting sched %d for %d steps", rank, client_index, schedule_length);

        // Run the scheduled work
        int total_steps = 0;
        while (total_steps < schedule_length)
        {
            // Reset the hidden and cell states if the environment is done.
            if (agent.done)
            {
                agent.hx = torch::zeros({1, 512}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
                agent.cx = torch::zeros({1, 512}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            } else {
                // Detach the hidden and cell states from the computation graph.
                agent.hx = agent.hx.detach();
                agent.cx = agent.cx.detach();
            }

            // Move the model and the environment to the GPU if necessary.
            if (args.gpu_id >= 0)
            {
                agent.hx = agent.hx.to(torch::kCUDA);
                agent.cx = agent.cx.to(torch::kCUDA);
            }

            // Run the agent for a number of steps.
            for (int i = 0; i < args.a3c_steps; i++)
            {
                agent.action_train();
                total_steps += 1;

                if (agent.done)
                    break;
            }

            if (agent.done)
            {
                agent.state = agent.env.reset();
                log_debug("train %d terminated episode len %d", rank, agent.eps_len);
                agent.eps_len = 0;
            }

            // Initialize the discounted return tensor.
            torch::Tensor R = torch::zeros({1, 1}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            //R = R.detach();

            torch::IValue result;

            if (!agent.done)
            {
                // Compute the discounted return.
                result = agent.model.forward(torch::TensorList({agent.state.unsqueeze(0), agent.hx, agent.cx}));
                R = result.toTensorList().get(0).set_requires_grad(true);
            }

            // Move the discounted return tensor to the GPU if necessary.
            if (args.gpu_id >= 0)
                R = R.to(torch::kCUDA);

            agent.values.push_back(R.detach()); // possibly no detach

            // might not need autograd variable
            //torch::Tensor policy_loss = torch::autograd::Variable(torch::zeros({1}, torch::kFloat32));
            //torch::Tensor value_loss = torch::autograd::Variable(torch::zeros({1}, torch::kFloat32));
            //torch::Tensor gae = torch::autograd::Variable(torch::zeros({1}, torch::kFloat32));

            torch::Tensor policy_loss = torch::zeros({1}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            torch::Tensor value_loss = torch::zeros({1}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            torch::Tensor gae = torch::zeros({1, 1}, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));
            torch::Tensor delta, log_prob, value, adv;

            R = R.detach(); // possibly no detach

            // Walk through the trajectory in reverse order.
            for (int i = agent.rewards.size() - 1; i >= 0; i--)
            {
                // Compute the discounted return.
                R = args.gamma * R + agent.rewards[i];

                // Compute the value loss.
                value = agent.values[i];
                value_loss = value_loss + 0.5 * torch::pow(R - value, 2);

                // Compute the generalized advantage estimate.
                delta = agent.rewards[i] + args.gamma * agent.values[i + 1] - agent.values[i];
                gae = gae * args.gamma * args.tau + delta; // possibly no detach

                // Compute the policy loss.
                log_prob = agent.log_probs[i];
                policy_loss = policy_loss - log_prob * gae.detach();
                policy_loss = policy_loss - 0.45 * agent.entropies[i];
            }

            // Zero the gradients.
            //optimizer.zero_grad();
            agent.model.zero_grad();

            // Backpropagate the loss.
            torch::Tensor loss = policy_loss + 0.5f * value_loss;
            loss.backward();//sum().backward();

            //std::cout << "policy_loss grad " << policy_loss.grad() << std::endl;

            // Check if the model params are leaves
            if (!agent.model.parameters()[0].is_leaf())
                throw runtime_error("model params are not leaves");

            // Clip the gradients.
            torch::nn::utils::clip_grad_norm_(agent.model.parameters(), 40.0f); // TODO: make this a parameter

            // Update the model parameters.
            optimizer.step();

            // Clear the trajectory.
            agent.clear_actions();

            log_debug("train %d step %d loss %f grad %f", rank, total_steps, loss.sum().item<float>(), agent.model.parameters()[0].grad().sum().item<float>());
        }

        // Send the updated model parameters to the scheduler.
        sendInt(0, rank);
        sendInt(0, MSG_UPDATE_GLOBAL_MODEL);

        // Hack the agent model to find the delta
        agent.model.add(init_model, -1.0f);

        std::vector<char> delta_params = agent.model.serialize();
        sendInt(0, client_index);
        sendBuffer(0, delta_params);
    }

    return 0;
}
