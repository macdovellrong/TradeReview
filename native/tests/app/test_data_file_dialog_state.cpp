#include "tradereview/app/DataFileDialogState.h"
#include "tradereview/core/Assertions.h"

#include <QDir>
#include <QSettings>
#include <QTemporaryDir>

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_data_file_dialog_state_defaults_to_empty_directory()
{
    QTemporaryDir dir;
    tradereview::core::assert_true(dir.isValid(), "temporary settings directory");
    QSettings settings(dir.filePath("dialog.ini"), QSettings::IniFormat);

    tradereview::core::assert_true(
        tradereview::app::last_data_directory(settings).isEmpty(),
        "missing last data directory is empty");
}

void test_data_file_dialog_state_remembers_selected_file_directory()
{
    QTemporaryDir dir;
    tradereview::core::assert_true(dir.isValid(), "temporary settings directory");
    QSettings settings(dir.filePath("dialog.ini"), QSettings::IniFormat);
    const auto data_directory = QDir(dir.path()).filePath("datasets");
    const auto data_path = QDir(data_directory).filePath("sample.duckdb");

    tradereview::app::remember_data_file_directory(settings, data_path);

    tradereview::core::assert_equal(
        tradereview::app::last_data_directory(settings).toStdString(),
        data_directory.toStdString(),
        "remembered data directory");
}

struct RegisterDataFileDialogStateTests {
    RegisterDataFileDialogStateTests()
    {
        tradereview::tests::register_test(
            "data file dialog state defaults to empty directory",
            test_data_file_dialog_state_defaults_to_empty_directory);
        tradereview::tests::register_test(
            "data file dialog state remembers selected file directory",
            test_data_file_dialog_state_remembers_selected_file_directory);
    }
};

const RegisterDataFileDialogStateTests register_data_file_dialog_state_tests;

} // namespace
