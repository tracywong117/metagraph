#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip> // For formatting time

class ProgressBar {
private:
    int width;           // Width of the progress bar
    std::string title;   // Title of the progress bar
    int total;           // Total number of steps
    int current;         // Current progress count
    std::chrono::steady_clock::time_point startTime; // Start time of the progress bar

    // Helper function to render the bar
    void render() const {
        float progress = static_cast<float>(current) / total;
        int completedWidth = static_cast<int>(std::round(progress * width));

        // Calculate elapsed time and estimated remaining time
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        int estimatedTotalTime = static_cast<int>((elapsed / progress));
        int remainingTime = estimatedTotalTime - elapsed;

        // Render the title
        std::cout << "\r" << title << " [";

        // Render the progress bar
        for (int i = 0; i < width; ++i) {
            if (i < completedWidth) {
                std::cout << "#"; // Completed section
            } else {
                std::cout << " "; // Remaining section
            }
        }

        // Render percentage
        std::cout << "] " << static_cast<int>(progress * 100) << "%";

        // Render elapsed and remaining time
        std::cout << " | Elapsed: " << formatTime(elapsed)
                  << " | ETA: " << (current < total ? formatTime(remainingTime) : "00:00");

        // Flush the output to ensure it updates
        std::cout.flush();
    }

    // Helper function to format time in HH:MM:SS
    static std::string formatTime(int seconds) {
        int hrs = seconds / 3600;
        int mins = (seconds % 3600) / 60;
        int secs = seconds % 60;
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << hrs << ":"
            << std::setw(2) << std::setfill('0') << mins << ":"
            << std::setw(2) << std::setfill('0') << secs;
        return oss.str();
    }

public:
    // Constructor to initialize the progress bar
    ProgressBar(int width, const std::string& title, int total)
        : width(width), title(title), total(total), current(0),
          startTime(std::chrono::steady_clock::now()) {}

    // Increment the progress bar by 1
    void update() {
        if (current < total) {
            ++current;
            if (current % (total / width) == 0 || current == total) { // Update less frequently
                render();
            }
        }
    }

    // Finish the progress bar and close it
    void finish() {
        current = total;
        render();
        std::cout << std::endl; // Move to the next line
    }
};

#endif // PROGRESS_BAR_H