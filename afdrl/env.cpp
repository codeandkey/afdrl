/**
 * @file env.cpp
 * @brief Generic environment interface
 */

#include "env.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

using namespace ale;

AtariEnv::AtariEnv(const std::string &rom_path, EnvConfig config, int seed, bool display)
{
  ale = new ale::ALEInterface();

  if (seed == -1)
    seed = time(NULL);

  ale->setInt("random_seed", seed);
  ale->setBool("display_screen", display);
  ale->loadROM(rom_path);

  screen_width = ale->getScreen().width();
  screen_height = ale->getScreen().height();
  screen_channels = config.frame_stack;
  num_actions = ale->getMinimalActionSet().size();

  this->config = config;
  reset();
}

torch::Tensor AtariEnv::observe()
{
  // Get the screen data in full color
  vector<unsigned char> data;
  ale->getScreenRGB(data);

  // Convert the screen data to a cv::Mat
  cv::Mat image(screen_height, screen_width, CV_8UC3, data.data());

  // Convert the RGB to Y channel
  cv::Mat y_channel;
  cv::cvtColor(image, y_channel, cv::COLOR_RGB2GRAY);

  // Crop the image with the env config (left, top, right, bottom)
  cv::Mat cropped = y_channel(cv::Rect(config.crop_x, config.crop_y, config.crop_width, config.crop_height));

  // Resize the image to the desired size
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(80, 80),
             cv::INTER_LINEAR);

  // Apply binary threshold
  cv::threshold(resized, resized, 128, 255, cv::THRESH_BINARY);

  // Convert the cv::Mat to a torch tensor
  return torch::from_blob(resized.data, {1, 80, 80}, torch::TensorOptions().dtype(torch::kByte)).toType(torch::kFloat).div(255);
}

AtariEnv::~AtariEnv() { delete ale; }

torch::Tensor AtariEnv::reset() {
  ale->reset_game();

  // Clear the frame skip deque
  frame_stack_deque.clear();

  torch::Tensor obs = observe();

  // Initialize the frame skip deque with the initial screen
  for (int i = 0; i < config.frame_stack; i++) {
    frame_stack_deque.push_back(obs);
  }

  // Return the concatenated frames from the frame skip deque
  std::vector<torch::Tensor> frame_stack_deque_vec(frame_stack_deque.begin(),
                                                  frame_stack_deque.end());

  return torch::cat(frame_stack_deque_vec);
}

std::tuple<torch::Tensor, float, bool> AtariEnv::step(int action) {
  // Get the minimal action set
  const ActionVect &actions = ale->getMinimalActionSet();

  // Perform the action for the number of frame skips
  float reward = 0;

  for (int i = 0; i < config.frame_skip; i++) {
    reward += ale->act(actions[action]);

    frame_stack_deque.push_front(observe());
    frame_stack_deque.pop_back();
  }

  bool terminal = ale->game_over();

  // Return the concatenated frames from the frame skip deque
  std::vector<torch::Tensor> frame_stack_deque_vec(frame_stack_deque.begin(),
                                                  frame_stack_deque.end());
  return std::make_tuple(torch::cat(frame_stack_deque_vec), reward,
                         terminal);
}
