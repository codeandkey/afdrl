/**
 * @file agent.h
 *
 * @brief Agent class definition.
 */

#pragma once

#include <string>
#include <vector>

#include "torch_pch.h"

#include "model.h"
#include "env.h"
#include "args.h"

class Agent {
  public:
    /**
     * @brief Agent constructor.
     *
     * @param model The model to be used.
     * @param env The environment to be used.
     */
    Agent(LSTMModel& model, AtariEnv& env, Args args);

    /**
     * @brief Perform a training step.
     */
    void action_train();

    /**
     * @brief Perform a testing step.
     */
    void action_test();

    /**
     * Clear the action history.
     */
    void clear_actions();

  private:
    // Arguments
    Args args;

  public:

    // LSTM hx, cx state
    torch::autograd::Variable hx, cx;

    // Environment
    AtariEnv& env;

    // Model
    LSTMModel& model;

    // Current observation
    torch::Tensor state;

    // Log probabilities
    std::vector<torch::Tensor> log_probs;

    // Rewards
    std::vector<float> rewards;
    float reward = 0;

    // Episode length
    int eps_len = 0;
    
    // Values
    std::vector<torch::autograd::Variable> values;

    // Entropies
    std::vector<torch::Tensor> entropies;

    // Terminal state
    bool done = true;
};
