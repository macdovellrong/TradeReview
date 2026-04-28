#pragma once

#include <QWidget>

class QLabel;
class QString;

namespace tradereview::app {

class SideInfoPanelWidget final : public QWidget {
public:
    explicit SideInfoPanelWidget(QWidget* parent = nullptr);

    void setDatasetName(const QString& name);
    void setDataRange(const QString& range);
    void setVisibleRange(const QString& range);
    void setLayoutSummary(const QString& summary);
    void setReplaySummary(const QString& summary);
    void setLastMessage(const QString& message);
    void resetDataset();

private:
    QLabel* dataset_value_ = nullptr;
    QLabel* data_range_value_ = nullptr;
    QLabel* visible_range_value_ = nullptr;
    QLabel* layout_value_ = nullptr;
    QLabel* replay_value_ = nullptr;
    QLabel* message_value_ = nullptr;
};

} // namespace tradereview::app
