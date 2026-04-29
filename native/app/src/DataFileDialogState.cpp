#include "tradereview/app/DataFileDialogState.h"

#include <QFileInfo>
#include <QSettings>

namespace tradereview::app {
namespace {

constexpr auto kLastDataDirectoryKey = "data/last_dir";

} // namespace

QString last_data_directory(const QSettings& settings)
{
    return settings.value(kLastDataDirectoryKey).toString();
}

void remember_data_file_directory(QSettings& settings, const QString& data_file_path)
{
    if (data_file_path.isEmpty()) {
        return;
    }

    const auto directory = QFileInfo(data_file_path).absolutePath();
    if (!directory.isEmpty()) {
        settings.setValue(kLastDataDirectoryKey, directory);
    }
}

} // namespace tradereview::app
