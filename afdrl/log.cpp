#include "log.h"

#include <cstdarg>
#include <mutex>

static int log_debug_enabled = 0;
static std::mutex log_mutex;

void log_set_debug(int enabled)
{
    log_debug_enabled = enabled;
}

void log_info(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(log_mutex);
    va_list args;
    va_start(args, format);
    fprintf(stderr, "\033[0;32mINFO    \e\033[0m|\e\033[0;32m ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n\033[0m");
    va_end(args);
}

void log_warn(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(log_mutex);
    va_list args;
    va_start(args, format);
    fprintf(stderr, "\033[0;33mWARNING \e\033[0m|\e\033[0;33m ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n\033[0m");
    va_end(args);
}

void log_error(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(log_mutex);
    va_list args;
    va_start(args, format);
    fprintf(stderr, "\033[0;31mERROR   \e\033[0m|\e\033[0;31m ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n\033[0m");
    va_end(args);
}

void log_debug(const char* format, ...)
{
    if (!log_debug_enabled)
        return;

    std::lock_guard<std::mutex> lock(log_mutex);

    va_list args;
    va_start(args, format);
    fprintf(stderr, "\033[0;34m");
    fprintf(stderr, "\033[0;34mDEBUG   \e\033[0m|\e\033[0;34m ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n\033[0m");
    va_end(args);
}