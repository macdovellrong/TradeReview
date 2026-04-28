#pragma once

#include <QWidget>

class QLabel;
class QString;

namespace tradereview::app {

class StatusStripWidget final : public QWidget {
public:
    explicit StatusStripWidget(QWidget* parent = nullptr);

    void setStateText(const QString& text);
    void setDuckDbText(const QString& text);
    void setDataRangeText(const QString& text);
    void setRendererText(const QString& text);
    void setMessageText(const QString& text);

private:
    QLabel* state_label_ = nullptr;
    QLabel* duckdb_label_ = nullptr;
    QLabel* data_range_label_ = nullptr;
    QLabel* renderer_label_ = nullptr;
    QLabel* message_label_ = nullptr;
};

} // namespace tradereview::app
