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
    AtariEnv(const std::string& rom_path, bool display_screen, int frame_skip=3, int frame_stack=3, int max_episode_length=10000, int seed=-1);

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
    int get_num_actions() const;

    /**
     * Get the screen height.
     *
     * @return The screen height.
     */
    int get_screen_height() const;

    /**
     * Get the screen width.
     *
     * @return The screen width.
     */
    int get_screen_width() const;

    /**
     * Get the screen channels.
     *
     * @return The screen channels.
     */
    int get_screen_channels() const;

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
    // ALE environment
    ale::ALEInterface* ale;

    // Environment parameters
    int num_actions, screen_height, screen_width, screen_channels;
    int frame_skip, frame_stack, max_episode_length; 

    // Deque of recent observations
    std::deque<torch::Tensor> frame_stack_deque;
};


#endif
