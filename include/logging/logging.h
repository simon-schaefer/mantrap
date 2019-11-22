#ifndef LOGGING_H
#define LOGGING_H

#include <chrono>
#include <ctime>
#include <iostream>

template<typename T>
void LOG_INFO(const T& message)
{
    const std::chrono::time_point_cast<std::chrono::system_clock> stamp = std::chrono::system_clock::now();
    const std::time_t stamp_time = std::chrono::system_clock::to_time_t(stamp);
    std::cout << "[INFO " << std::ctime(&stamp_time) << "]: " << message << std::endl;
}


template<typename T>
void LOG_ERROR(const T& message)
{
    const std::chrono::time_point_cast<std::chrono::system_clock> stamp = std::chrono::system_clock::now();
    const std::time_t stamp_time = std::chrono::system_clock::to_time_t(stamp);
    std::cerr << "[ERROR " << std::ctime(&stamp_time) << "]: " << message << std::endl;
}


#endif //LOGGING_H
