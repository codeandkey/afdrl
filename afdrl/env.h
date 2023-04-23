/**
 * @file env.h
 * @brief Generic environment interface
 */

#ifndef AFDRL_ENV_H
#define AFDRL_ENV_H

#include "torch_pch.h"
#include <ale/ale_interface.hpp>
#include <deque>
#include <vector>

struct EnvConfig
{
  int frame_skip = 3;
  int frame_stack = 3;
  int max_episode_length = 10000;

  int crop_x = 0;
  int crop_y = 0;
  int crop_width = 0;
  int crop_height = 0;
};

/**
 * This class contains an ALE environment and provides a generic interface to
 * interact with it.
 */
class AtariEnv {
public:
    /**
     * Constructs an Atari environment.
     * 
     * @param rom_path The path to the ROM file.
     * @param display_screen Whether to display the screen.
     * @param frame_skip The number of frames to skip.
     * @param max_episode_length The maximum number of steps per episode.
     */
    AtariEnv(const std::string& rom_path, EnvConfig config, int seed=-1, bool display=false);

    /**
     * Destructs the Atari environment.
     */
    ~AtariEnv();

    /**
     * Resets the environment.
     * 
     * @return The initial state.
     */
    torch::Tensor reset();

    /**
     * Steps the environment.
     * 
     * @param action The action to take.
     * @return The next state, reward received, and terminal state.
     */
    std::tuple<torch::Tensor, float, bool> step(int action);

    /**
     * Get the number of actions.
     *
     * @return The number of actions.
     */
    int get_num_actions() const { return num_actions; }

    /**
     * Get the screen height.
     *
     * @return The screen height.
     */
    int get_screen_height() const { return screen_height; }

    /**
     * Get the screen width.
     *
     * @return The screen width.
     */
    int get_screen_width() const { return screen_width; }

    /**
     * Get the screen channels.
     *
     * @return The screen channels.
     */
    int get_screen_channels() const { return screen_channels; }

    /**
     * Serializes the environment state into a buffer.
     * 
     * @return The buffer containing the serialized environment state.
     */
    std::vector<char> serialize() const;

    /**
     * Deserializes the environment state from a buffer.
     * 
     * @param buffer The buffer containing the serialized environment state.
     */
    void deserialize(const std::vector<char>& buffer);

private:
    /**
     * Observes the environment.
     * 
     * @return The current state.
     */
    torch::Tensor observe();

    // Configuration
    EnvConfig config;

    // ALE environment
    ale::ALEInterface* ale;

    // Environment parameters
    int num_actions, screen_height, screen_width, screen_channels;

    // Deque of recent observations
    std::deque<torch::Tensor> frame_stack_deque;
};


#endif
