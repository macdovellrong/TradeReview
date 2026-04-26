#include "tradereview/app/ErrorPresenter.h"

#include <QMessageBox>
#include <QStatusBar>

namespace tradereview::app {
namespace {

QString qstring_from_std(const std::string& text)
{
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

QString data_error_kind(data::DataErrorCode code)
{
    switch (code) {
    case data::DataErrorCode::FileNotFound:
        return "File not found";
    case data::DataErrorCode::OpenFailed:
        return "Open failed";
    case data::DataErrorCode::SchemaMismatch:
        return "Schema mismatch";
    case data::DataErrorCode::MissingTable:
        return "Missing table";
    case data::DataErrorCode::MissingColumn:
        return "Missing column";
    case data::DataErrorCode::QueryFailed:
        return "Query failed";
    case data::DataErrorCode::InvalidRequest:
        return "Invalid request";
    case data::DataErrorCode::None:
        return "Data error";
    }
    return "Data error";
}

void show_message(QWidget* parent, const ErrorMessage& message, bool show_dialog)
{
    if (!show_dialog) {
        return;
    }
    QMessageBox::warning(parent, message.title, message.detail);
}

void show_status(QStatusBar* status_bar, const ErrorMessage& message)
{
    if (status_bar != nullptr) {
        status_bar->showMessage(message.status);
    }
}

} // namespace

ErrorMessage build_error_message(const QString& context, const data::DataError& error)
{
    const auto title = context + " failed";
    QString detail = data_error_kind(error.code);
    if (!error.message.empty()) {
        detail += ": " + qstring_from_std(error.message);
    }
    if (!error.path.empty()) {
        detail += "\nFile: " + qstring_from_std(error.path);
    }
    if (!error.table.empty()) {
        detail += "\nTable: " + qstring_from_std(error.table);
    }

    return ErrorMessage{
        title,
        detail,
        title + ": " + detail.section('\n', 0, 0),
    };
}

ErrorMessage build_error_message(const QString& context, const std::exception& error)
{
    const auto title = context + " failed";
    const auto detail = QString::fromUtf8(error.what());
    return ErrorMessage{
        title,
        detail,
        title + ": " + detail,
    };
}

void present_error(
    QWidget* parent,
    QStatusBar* status_bar,
    const QString& context,
    const data::DataError& error,
    bool show_dialog)
{
    const auto message = build_error_message(context, error);
    show_status(status_bar, message);
    show_message(parent, message, show_dialog);
}

void present_error(
    QWidget* parent,
    QStatusBar* status_bar,
    const QString& context,
    const std::exception& error,
    bool show_dialog)
{
    if (const auto* data_exception = dynamic_cast<const data::DataException*>(&error); data_exception != nullptr) {
        present_error(parent, status_bar, context, data_exception->error(), show_dialog);
        return;
    }

    const auto message = build_error_message(context, error);
    show_status(status_bar, message);
    show_message(parent, message, show_dialog);
}

} // namespace tradereview::app
