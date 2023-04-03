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

Mat aleToImage(ale::ALEInterface &ale) {
  // Get the screen from the ALE environment
  const ale::ALEScreen &screen = ale.getScreen();

  // Get the screen dimensions
  int height = screen.height();
  int width = screen.width();

  // Convert the screen to a Mat
  return Mat(height, width, CV_8UC3, screen.getArray());
}

// Process an image from an ALE environment, rescaling it to 1 channel, 80 x 80
torch::Tensor imageToTensor(const Mat &image) {
  // Convert the image to grayscale
  Mat gray_image;
  cvtColor(image, gray_image, COLOR_BGR2GRAY);

  // Resize the image to 80 x 80
  Mat resized_image;
  resize(gray_image, resized_image, Size(80, 80));

  // Convert the image to a tensor
  torch::Tensor tensor_image =
      torch::from_blob(resized_image.data, {1, 80, 80}, torch::kByte);

  // Normalize the image
  tensor_image = tensor_image.toType(torch::kFloat);
  tensor_image = tensor_image.div(255);

  return tensor_image;
}

using namespace ale;

AtariEnv::AtariEnv(const std::string &rom_path, bool display_screen,
                   int frame_skip, int frame_stack, int max_episode_length) {
  ale = new ale::ALEInterface();
  ale->setBool("display_screen", display_screen);
  ale->loadROM(rom_path);

  // Get the screen shape parameters from the environment
  screen_width = 80; // fixed by the image resize routines above
  screen_height = 80;
  screen_channels = 1;
  num_actions = ale->getMinimalActionSet().size();

  this->frame_skip = frame_skip;
  this->frame_stack = frame_stack;
  this->max_episode_length = max_episode_length;

  reset();
}

AtariEnv::~AtariEnv() { delete ale; }

torch::Tensor AtariEnv::reset() {
  ale->reset_game();

  // Clear the frame skip deque
  frame_stack_deque.clear();

  // Get the initial screen from the environment
  Mat image = aleToImage(*ale);

  // Initialize the frame skip deque with the initial screen
  for (int i = 0; i < frame_stack; i++) {
    frame_stack_deque.push_back(imageToTensor(image));
  }

  // Return the concatenated frames from the frame skip deque
  std::vector<torch::Tensor> frame_stack_deque_vec(frame_stack_deque.begin(),
                                                  frame_stack_deque.end());
  return torch::cat(frame_stack_deque_vec, -1);
}

std::tuple<torch::Tensor, float, bool> AtariEnv::step(int action) {
  // Get the minimal action set
  const ActionVect &actions = ale->getMinimalActionSet();

  // Perform the action for the number of frame skips
  float reward = 0;

  for (int i = 0; i < frame_skip; i++) {
    reward += ale->act(actions[action]);
  }

  bool terminal = ale->game_over();

  // Get the new screen from the environment
  Mat image = aleToImage(*ale);

  // Add the new frame to the frame skip deque
  frame_stack_deque.push_back(imageToTensor(image));
  frame_stack_deque.pop_front();

  // Return the concatenated frames from the frame skip deque
  std::vector<torch::Tensor> frame_stack_deque_vec(frame_stack_deque.begin(),
                                                  frame_stack_deque.end());
  return std::make_tuple(torch::cat(frame_stack_deque_vec, -1), reward,
                         terminal);
}

int AtariEnv::get_num_actions() const { return num_actions; }

int AtariEnv::get_screen_height() const { return screen_height; }

int AtariEnv::get_screen_width() const { return screen_width; }

int AtariEnv::get_screen_channels() const { return screen_channels; }

vector<char> AtariEnv::serialize() const {
  // Serialize the ALE system state
  ale::ALEState system_state = ale->cloneSystemState();
  std::string ale_state_str = system_state.serialize();

  vector<char> state;

  // Get the ALE state string length
  int ale_state_str_len = ale_state_str.length();
  int frame_size = frame_stack_deque[0].numel() * sizeof(float);

  // Reserve space for the state vector
  state.reserve(sizeof(int) + ale_state_str_len + sizeof(float) +
                frame_stack_deque.size() * frame_size);

  // Write the length of the ALE state string
  memcpy(state.data(), &ale_state_str_len, sizeof(int));

  // Write the ALE state string (without the null terminator)
  memcpy(state.data() + sizeof(int), ale_state_str.c_str(), ale_state_str_len);

  // Write the frame skip deque
  for (int i = 0; i < frame_skip; i++) {
    // Get the frame data
    float *frame_data = frame_stack_deque[i]
                            .to(torch::kFloat)
                            .contiguous()
                            .view({-1})
                            .to(torch::kCPU)
                            .data_ptr<float>();

    // Write frame data to the state vector
    memcpy(state.data() + sizeof(int) + ale_state_str_len + i * frame_size,
           frame_data, frame_size);
  }

  return state;
}

void AtariEnv::deserialize(const vector<char> &state) {
  // Read the length of the ALE state string
  int ale_state_str_len = *(int *)state.data();

  // Read the ALE state string
  string ale_state_str(state.data() + sizeof(int), ale_state_str_len);

  // Deserialize the ALE system state
  ale::ALEState system_state(ale_state_str);
  ale->restoreSystemState(system_state);

  // Clear the frame skip deque
  frame_stack_deque.clear();

  // Deserialize the frame skip deque
  int frame_size = screen_height * screen_width;
  for (int i = 0; i < frame_stack; i++) {
    torch::Tensor frame =
        torch::from_blob((void *)(state.data() + sizeof(int) +
                                  ale_state_str_len + i * frame_size),
                         {1, screen_height, screen_width}, torch::kFloat);
    frame_stack_deque.push_back(frame);
  }
}
