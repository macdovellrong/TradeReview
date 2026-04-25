#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tradereview::tests {

struct TestCase {
    std::string name;
    std::function<void()> run;
};

std::vector<TestCase>& registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

void register_test(std::string name, std::function<void()> run)
{
    registry().push_back(TestCase{std::move(name), std::move(run)});
}

} // namespace tradereview::tests

int main()
{
    int failures = 0;
    for (const auto& test : tradereview::tests::registry()) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        return 1;
    }
    std::cout << tradereview::tests::registry().size() << " native test(s) passed\n";
    return 0;
}
