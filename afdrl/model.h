#pragma once

// remove me
#include <iostream>

#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
#include <tuple>

#include "env.h"
#include "torch_pch.h"

class LSTMModel : public torch::nn::Module {
public:
  LSTMModel(int channels, int n_actions) {
    this->n_actions = n_actions;

    torch::nn::Conv2dOptions options1(channels, 32, 5);
    options1.stride(1);
    options1.padding(2);

    conv1 = torch::nn::Conv2d(options1);
    maxp1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2));

    torch::nn::Conv2dOptions options2(32, 32, 5);
    options2.stride(1);
    options2.padding(1);

    conv2 = torch::nn::Conv2d(options2);
    maxp2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2));

    torch::nn::Conv2dOptions options3(32, 64, 4);
    options3.stride(1);
    options3.padding(1);

    conv3 = torch::nn::Conv2d(options3);
    maxp3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2));

    torch::nn::Conv2dOptions options4(64, 64, 3);
    options4.stride(1);
    options4.padding(1);

    conv4 = torch::nn::Conv2d(options4);
    maxp4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2));

    lstm = torch::nn::LSTMCell(torch::nn::LSTMCellOptions(1024, 512));

    // Initialize the actor and critic linear layers.
    actor_linear = torch::nn::Linear(512, n_actions);
    critic_linear = torch::nn::Linear(512, 1);

    // Initialize the actor and critic weights with Normal distribution of mean
    // 0 and variance 1.
    torch::nn::init::normal_(actor_linear->weight, 0, 1);
    torch::nn::init::normal_(critic_linear->weight, 0, 1);

    // Initialize the actor and critic biases with 0.
    torch::nn::init::constant_(actor_linear->bias, 0);
    torch::nn::init::constant_(critic_linear->bias, 0);

    // Register the actor and critic linear layers as submodules.
    register_module("actor_linear", actor_linear);
    register_module("critic_linear", critic_linear);

    // Register the convolutional layers as submodules.
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);

    // Register the max pooling layers as submodules.
    register_module("maxp1", maxp1);
    register_module("maxp2", maxp2);
    register_module("maxp3", maxp3);
    register_module("maxp4", maxp4);

    // Register the LSTM layer as a submodule.
    register_module("lstm", lstm);

    /*std::cout << "MODEL init param requires grad: "
              << actor_linear->weight.requires_grad() << std::endl;*/
  }

  /**
   * Add weighted parameters from another model.
   *
   * @param other The other model.
   * @param tau The weight of the other model.
   */
  void add(const LSTMModel &other, float tau) {
    auto sum_params = [](const LSTMModel &md) -> float {
      float sum = 0;
      for (auto &param : md.parameters()) {
        sum += param.sum().item<float>();
      }
      return sum;
    };

    // Add the convolutional layers.
    addConv(conv1, other.conv1, tau);
    addConv(conv2, other.conv2, tau);
    addConv(conv3, other.conv3, tau);
    addConv(conv4, other.conv4, tau);

    // Add the actor and critic linear layers.
    addLinear(actor_linear, other.actor_linear, tau);
    addLinear(critic_linear, other.critic_linear, tau);
  }

  /**
   * Add weighted parameters from another convolutional layer.
   *
   * @param layer The layer to add to.
   * @param other The other layer.
   * @param tau The weight of the other layer.
   */
  void addConv(torch::nn::Conv2d layer, torch::nn::Conv2d other, float tau) {
    layer->weight.requires_grad_(false);
    layer->weight += tau * other->weight;
    layer->weight.detach_();
    layer->weight.requires_grad_(true);
    layer->bias.requires_grad_(false);
    layer->bias += tau * other->bias;
    layer->bias.detach_();
    layer->bias.requires_grad_(true);
  }

  /**
   * Add weighted parameters from another linear layer.
   *
   * @param layer The layer to add to.
   * @param other The other layer.
   * @param tau The weight of the other layer.
   */
  void addLinear(torch::nn::Linear layer, torch::nn::Linear other, float tau) {
    layer->weight.requires_grad_(false);
    layer->weight += tau * other->weight;
    layer->weight.detach_();
    layer->weight.requires_grad_(true);
    layer->bias.requires_grad_(false);
    layer->bias += tau * other->bias;
    layer->bias.detach_();
    layer->bias.requires_grad_(true);
  }

  void print() {
    for (auto &param : named_parameters())
      std::cout << param.key() << " = " << param.value().sum().item()
                << std::endl;
  }

  /**
   * Serializes the model to a vector of bytes.
   *
   * @return std::vector<char> The serialized model.
   */
  std::vector<char> serialize() {
    std::vector<char> buffer;
    std::stringstream stream;

    torch::serialize::OutputArchive output_archive;

    for (auto &param : named_parameters()) {
      output_archive.write(param.key(), param.value());
      /*std::cout << "serialize param " << param.key() << " = "
                << param.value().sum().item() << std::endl;*/
    }

    // Serialize the model to a string stream.
    output_archive.save_to(stream);

    // Convert the string stream to a vector of bytes.
    buffer = std::vector<char>((std::istreambuf_iterator<char>(stream)),
                               std::istreambuf_iterator<char>());

    return buffer;
  }

  /**
   * Deserializes the model from a vector of bytes.
   *
   * @param buffer The serialized model.
   */
  void deserialize(std::vector<char> buffer) {
    // std::cout << "Deserializing model..." << std::endl;
    // std::cout << "Buffer size: " << buffer.size() << std::endl;

    std::stringstream stream;

    // Convert the vector of bytes to a string stream.
    stream = std::stringstream(std::string(buffer.begin(), buffer.end()));

    torch::serialize::InputArchive stored_params;
    stored_params.load_from(stream);

    for (auto &param : named_parameters()) {
      torch::Tensor t;
      torch::Tensor current = param.value().clone();

      if (stored_params.try_read(param.key(), t)) {
        // param.value().requires_grad_(false);
        // param.value().copy_(t);
        // param.value().requires_grad_(true);
        // param.value() = t;
        // named_parameters()[param.key()] = t;
        // register_parameter(param.key(), t);
        param.value().requires_grad_(false);
        param.value() *= 0;
        param.value() += t;
        param.value().detach_();
        param.value().requires_grad_(true);
      } else {
        throw std::runtime_error("Could not find param: " + param.key());
      }

      /*std::cout << "deserialize param " << param.key() << " current "
                << current.sum().item() << " stored " << t.sum().item()
                << " new " << param.value().sum().item() << " new named "
                << named_parameters()[param.key()].sum().item() << std::endl;*/
    }
  }

  /**
   * Forward pass of the model.
   *
   * @param iv The input tensor.
   * @return torch::IValue The output tensor.
   */
  torch::IValue forward(torch::IValue iv) {
    auto lst = iv.toTensorList();
    torch::Tensor inputs = lst[0], hx = lst[1], cx = lst[2];

    // Pass the input through each convolutional layer, followed by a max
    // pooling layer.
    inputs = torch::relu(conv1->forward(inputs));
    inputs = maxp1->forward(inputs);
    inputs = torch::relu(conv2->forward(inputs));
    inputs = maxp2->forward(inputs);
    inputs = torch::relu(conv3->forward(inputs));
    inputs = maxp3->forward(inputs);
    inputs = torch::relu(conv4->forward(inputs));
    inputs = maxp4->forward(inputs);

    // Reshape the input to be 1 x 1 x 1024 (required by LSTM).
    inputs = inputs.view({inputs.size(0), -1});

    // Pass the input through the LSTM layer.
    auto lstm_out = lstm->forward(inputs, std::make_tuple(hx, cx));

    hx = std::get<0>(lstm_out);
    cx = std::get<1>(lstm_out);

    // Pass the first output from the LSTM through the actor and critic linear
    // layers.
    auto actor_out = actor_linear->forward(hx);
    auto critic_out = critic_linear->forward(hx);

    return torch::List<torch::Tensor>({critic_out, actor_out, hx, cx});
  }

  // Declare an actor and critic linear layer, and initialize them to nullptr.
  torch::nn::Linear actor_linear = nullptr, critic_linear = nullptr;

  // Declare 4 convolutional layers, 4 max pooling layers, 1 LSTM layer,
  // and initialize them to nullptr.
  torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr, conv3 = nullptr,
                    conv4 = nullptr; // 32, 64, 64, 64
  torch::nn::MaxPool2d maxp1 = nullptr, maxp2 = nullptr, maxp3 = nullptr,
                       maxp4 = nullptr; // 2, 2, 2, 2
  torch::nn::LSTMCell lstm = nullptr;   // 1024, 512

  int n_actions; // The number of actions the agent can take.
};
