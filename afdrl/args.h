#pragma once

#include <string>
#include <iostream>

/**
 * Arguments data structure.
 */
struct Args {
  /**
   * Parse arguments from the command line.
   */
  Args(int argc, char** argv)
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        help = true;
      } else if (arg == "-l" || arg == "--log") {
        log_file = argv[++i];
      } else if (arg == "-r" || arg == "--results") {
        results_file = argv[++i];
      } else if (arg == "-g" || arg == "--gpu") {
        gpu_id = std::stoi(argv[++i]);
      } else if (arg == "-e" || arg == "--env") {
        env_name = argv[++i];
      } else if (arg == "-t" || arg == "--test") {
        test_steps = std::stoi(argv[++i]);
      } else if (arg == "-f" || arg == "--frame-skip") {
        frame_skip = std::stoi(argv[++i]);
      } else if (arg == "-m" || arg == "--max-episode-length") {
        max_episode_length = std::stoi(argv[++i]);
      } else if (arg == "-d" || arg == "--display-test") {
        display_test = true;
      } else if (arg == "-s" || arg == "--seed") {
        seed = std::stoi(argv[++i]);
      } else if (arg == "--min-offline") {
        min_offline = std::stoi(argv[++i]);
      } else if (arg == "--max-offline") {
        max_offline = std::stoi(argv[++i]);
      } else if (arg == "--min-offline-time") {
        min_offline_time = std::stoi(argv[++i]);
      } else if (arg == "--max-offline-time") {
        max_offline_time = std::stoi(argv[++i]);
      } else if (arg == "-c" || arg == "--num-clients") {
        num_clients = std::stoi(argv[++i]);
      } else if (arg == "--lr") {
        lr = std::stof(argv[++i]);
      } else if (arg == "--gamma") {
        gamma = std::stof(argv[++i]);
      } else if (arg == "--tau") {
        tau = std::stof(argv[++i]);
      } else if (arg == "--frame-stack") {
        frame_stack = std::stoi(argv[++i]);
      } else if (arg == "--num-steps") {
        num_steps = std::stoi(argv[++i]);
      } else {
        std::cout << "Unknown argument: " << arg << std::endl;
        usage(argv);
        exit(1);
      }
    }
  }

  /**
   * Writes usage information to stdout.
   */
  void usage(char** argv)
  {
    std::cout << "Usage: " << argv[0] << "  [options]" << std::endl;
    std::cout << "Options:" << std::endl;

    std::cout << "\t-h, --help" << std::endl;
    std::cout << "\t\tDisplay this help message." << std::endl;

    std::cout << "\t-l, --log-file" << std::endl;
    std::cout << "\t\tLog file." << std::endl;

    std::cout << "\t-r, --results-file" << std::endl;
    std::cout << "\t\tResults file." << std::endl;

    std::cout << "\t-g, --gpu" << std::endl;
    std::cout << "\t\tGPU ID (-1 = CPU)." << std::endl;

    std::cout << "\t-e, --env" << std::endl;
    std::cout << "\t\tEnvironment name." << std::endl;

    std::cout << "\t-t, --test" << std::endl;
    std::cout << "\t\tNumber of test steps." << std::endl;

    std::cout << "\t-f, --frame-skip" << std::endl;
    std::cout << "\t\tNumber of frames to skip between actions." << std::endl;

    std::cout << "\t-m, --max-episode-length" << std::endl;
    std::cout << "\t\tMaximum episode length (0 = no limit)." << std::endl;

    std::cout << "\t-d, --display-test" << std::endl;
    std::cout << "\t\tRender test episodes." << std::endl;

    std::cout << "\t-s, --seed" << std::endl;
    std::cout << "\t\tRandom seed." << std::endl;

    std::cout << "\t-o, --min-offline" << std::endl;
    std::cout << "\t\tMinimum number of offline steps." << std::endl;

    std::cout << "\t-p, --max-offline" << std::endl;
    std::cout << "\t\tMaximum number of offline steps." << std::endl;

    std::cout << "\t-c, --num-clients" << std::endl;
    std::cout << "\t\tNumber of simulated clients." << std::endl;

    std::cout << "\t--lr" << std::endl;
    std::cout << "\t\tLearning rate." << std::endl;

    std::cout << "\t--num-steps" << std::endl;
    std::cout << "\t\tNumber of steps per client per update." << std::endl;

    std::cout << "\t--frame-stack" << std::endl;
    std::cout << "\t\tNumber of frames to stack in model input." << std::endl;

    std::cout << "\t--gamma" << std::endl;
    std::cout << "\t\tDiscount factor." << std::endl;

    std::cout << "\t--tau" << std::endl;
    std::cout << "\t\tSoft update factor." << std::endl;

    std::cout << "\t--min-offline-time" << std::endl;
    std::cout << "\t\tMinimum number of offline global time steps per client." << std::endl;

    std::cout << "\t--max-offline-time" << std::endl;
    std::cout << "\t\tMaximum number of offline global time steps per client." << std::endl;

    std::cout << std::endl;
  }

  bool help = false; // Display help

  std::string log_file = "log.txt"; // Log file
  std::string results_file = "results.txt"; // Results file

  int gpu_id = -1; // -1 = CPU, 0 = first GPU, 1 = second GPU, etc.
  std::string env_name = "../roms/pong.bin"; // Environment name

  int test_steps = 100; // Number of test steps
  int frame_skip = 4; // Number of frames to skip between actions
  int frame_stack = 1; // Number of frames to stack in model input
  int max_episode_length = 10000; // 0 = no limit
  bool display_test = false; // Render test episodes
  int seed = 0; // Random seed

  int min_offline = 20; // Minimum number of offline steps
  int max_offline = 150; // Maximum number of offline steps
  int min_offline_time = 1; // Minimum number of offline global time steps
  int max_offline_time = 10; // Maximum number of offline global time steps

  int num_clients = 64; // Number of simulated clients
  int num_steps = 20; // Number of steps per client per update

  float lr = 0.0001; // Learning rate

  float gamma = 0.99; // Discount factor
  float tau = 1.0; // GAE factor
};
