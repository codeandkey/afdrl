#pragma once

void log_set_debug(int enabled=1);

void log_info(const char* format, ...);
void log_warn(const char* format, ...);
void log_error(const char* format, ...);
void log_debug(const char* format, ...);