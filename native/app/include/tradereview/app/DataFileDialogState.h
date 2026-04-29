#pragma once

#include <QString>

class QSettings;

namespace tradereview::app {

[[nodiscard]] QString last_data_directory(const QSettings& settings);
void remember_data_file_directory(QSettings& settings, const QString& data_file_path);

} // namespace tradereview::app
