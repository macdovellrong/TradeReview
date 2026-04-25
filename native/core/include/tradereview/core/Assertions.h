#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tradereview::core {

inline void assert_true(bool value, const std::string& message)
{
    if (!value) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename U>
void assert_equal(const T& actual, const U& expected, const std::string& message)
{
    if (!(actual == expected)) {
        std::ostringstream out;
        out << message << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(out.str());
    }
}

inline void assert_near(double actual, double expected, double tolerance, const std::string& message)
{
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << message << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(out.str());
    }
}

} // namespace tradereview::core
