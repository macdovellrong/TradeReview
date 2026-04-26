#include "tradereview/app/ErrorPresenter.h"

#include "tradereview/core/Assertions.h"
#include "tradereview/data/DataError.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_error_presenter_formats_data_errors_with_context()
{
    tradereview::data::DataError error;
    error.code = tradereview::data::DataErrorCode::MissingColumn;
    error.message = "missing columns: close volume";
    error.path = "C:/data/xau.duckdb";
    error.table = "candles_1m";

    const auto message = tradereview::app::build_error_message("Load Data", error);

    tradereview::core::assert_true(message.title.contains("Load Data"), "title includes context");
    tradereview::core::assert_true(message.detail.contains("missing columns: close volume"), "detail includes data message");
    tradereview::core::assert_true(message.detail.contains("C:/data/xau.duckdb"), "detail includes path");
    tradereview::core::assert_true(message.detail.contains("candles_1m"), "detail includes table");
    tradereview::core::assert_true(message.status.contains("Load Data failed"), "status is concise and contextual");
}

void test_error_presenter_formats_standard_exceptions()
{
    const auto message = tradereview::app::build_error_message(
        "Window reload",
        std::runtime_error("no dataset loaded"));

    tradereview::core::assert_true(message.title.contains("Window reload"), "exception title includes context");
    tradereview::core::assert_true(message.detail.contains("no dataset loaded"), "exception detail includes what");
    tradereview::core::assert_true(message.status.contains("Window reload failed"), "exception status includes context");
}

struct RegisterErrorPresenterTests {
    RegisterErrorPresenterTests()
    {
        tradereview::tests::register_test(
            "error presenter formats data errors with context",
            test_error_presenter_formats_data_errors_with_context);
        tradereview::tests::register_test(
            "error presenter formats standard exceptions",
            test_error_presenter_formats_standard_exceptions);
    }
};

const RegisterErrorPresenterTests register_error_presenter_tests;

} // namespace
