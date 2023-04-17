#include "agent.h"

#include <iostream>
#include <torch/serialize.h>

using namespace std;

Agent::Agent(LSTMModel& model, AtariEnv& env, Args args)
  : model(model), env(env), args(args) {
  state = env.reset();
}

void Agent::action_test()
{
  auto st = state.unsqueeze(0);

  // If the environment is done, reset the cx and hx tensors to 0.
  if (done)
  {
    hx = torch::zeros({1, 512}, torch::requires_grad());
    cx = torch::zeros({1, 512}, torch::requires_grad());
  } else {
    hx = hx.detach();
    cx = cx.detach();
  }

  // If the gpu ID is set, move the hx and cx tensors to the gpu.
  if (args.gpu_id >= 0)
  {
    hx = hx.to(torch::kCUDA);
    cx = cx.to(torch::kCUDA);
    st = st.to(torch::kCUDA);
  }

  // Get the value, logit, and (hx, cx) tensors from the model.
  auto output = model.forward(torch::TensorList({st, hx, cx})).toTensorList();

  // Get the value, logit, and new (hx, cx) tensors from the output.
  auto value = output.get(0);
  auto logit = output.get(1);
  hx = output.get(2);
  cx = output.get(3);

  // Get the probability distribution from the logit tensor.
  auto prob = torch::softmax(logit, 1);
  
  // Get the greedy action from the probability distribution.
  int action = torch::argmax(prob, 1).item<int64_t>();

  //std::cerr << "policy " << prob << " action " << action << std::endl;

  // Step the environment
  auto result = env.step(action);

  // Get the new state, reward, and done tensors from the result.
  state = std::get<0>(result);
  reward = std::get<1>(result);
  done = std::get<2>(result);

  // Move the state tensor to the GPU if the gpu ID is set.
  if (args.gpu_id >= 0)
  {
    state = state.to(torch::kCUDA);
  }

  // Increment the episode length
  ++eps_len;
}

void Agent::action_train()
{
  auto st = state.unsqueeze(0);

  // If the gpu ID is set, move the hx and cx tensors to the gpu.
  if (args.gpu_id >= 0)
  {
    hx = hx.to(torch::kCUDA);
    cx = cx.to(torch::kCUDA);
    st = st.to(torch::kCUDA);
  }

  // Get the value, logit, and (hx, cx) tensors from the model.
  auto output = model.forward(torch::TensorList({st, hx, cx})).toTensorList();

  // Get the value, logit, and new (hx, cx) tensors from the output.
  auto value = output.get(0);
  auto logit = output.get(1);
  hx = output.get(2);
  cx = output.get(3);

  values.push_back(value);

  // Get the probability distribution from the logit tensor.
  auto prob = torch::softmax(logit, 1);
  auto log_prob = torch::log_softmax(logit, 1);

  auto action = prob.multinomial(1).data();

  log_prob = log_prob.gather(1, action.detach());
  log_probs.push_back(log_prob);

  // Get the entropy from the prob and log_prob tensors.
  auto entropy = -(prob * log_prob).sum(1);
  entropies.push_back(entropy);
  
  // Sample an action from the probability distribution.
  // Step the environment
  auto result = env.step(action.item().toInt());

  // Get the new state, reward, and done tensors from the result.
  state = std::get<0>(result);
  reward = std::get<1>(result);
  done = std::get<2>(result);

  // Bound the reward between -1 and 1.
  reward = std::max(-1.0f, std::min(reward, 1.0f));
  rewards.push_back(reward);

  // Move the state tensor to the GPU if the gpu ID is set.
  if (args.gpu_id >= 0)
  {
    state = state.to(torch::kCUDA);
  }

  // Increment the episode length
  ++eps_len;
}

void Agent::clear_actions()
{
  values.clear();
  log_probs.clear();
  entropies.clear();
  rewards.clear();
}
