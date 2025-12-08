#pragma once
#include <chrono>
#include <string>
#include <iostream>

// Use steady_clock for stable interval measurement
class Timer {
public:
    Timer(std::string name) : start_time(std::chrono::steady_clock::now()), timer_name(name) {}

    double stop() {
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        std::cout << "Elapsed time for " << timer_name << ": " << duration.count() << " milliseconds\n";
        return duration.count();
    }

    ~Timer() {
        stop();
    }

private:
    std::chrono::steady_clock::time_point start_time;
    std::string timer_name;
};
