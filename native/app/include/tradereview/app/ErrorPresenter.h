#pragma once

#include "tradereview/data/DataError.h"

#include <QString>

#include <exception>

class QStatusBar;
class QWidget;

namespace tradereview::app {

struct ErrorMessage {
    QString title;
    QString detail;
    QString status;
};

[[nodiscard]] ErrorMessage build_error_message(const QString& context, const data::DataError& error);
[[nodiscard]] ErrorMessage build_error_message(const QString& context, const std::exception& error);
void present_error(
    QWidget* parent,
    QStatusBar* status_bar,
    const QString& context,
    const data::DataError& error,
    bool show_dialog = true);
void present_error(
    QWidget* parent,
    QStatusBar* status_bar,
    const QString& context,
    const std::exception& error,
    bool show_dialog = true);

} // namespace tradereview::app
