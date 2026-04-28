#include "tradereview/app/SideInfoPanelWidget.h"

#include "tradereview/app/AppTheme.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QString>
#include <QVBoxLayout>

namespace tradereview::app {
namespace {

QLabel* addRow(QVBoxLayout& layout, QWidget& parent, const QString& label)
{
    auto* row = new QWidget(&parent);
    auto* row_layout = new QHBoxLayout(row);
    row_layout->setContentsMargins(0, 3, 0, 3);
    row_layout->setSpacing(8);

    auto* name = new QLabel(label, row);
    name->setStyleSheet("color: #8fa0b8;");
    auto* value = new QLabel("--", row);
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value->setWordWrap(true);

    row_layout->addWidget(name);
    row_layout->addWidget(value, 1);
    layout.addWidget(row);
    return value;
}

} // namespace

SideInfoPanelWidget::SideInfoPanelWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("SideInfoPanelWidget");
    setFixedWidth(theme::Size::SidePanelWidth);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(12, 10, 12, 10);
    root->setSpacing(8);

    auto* title = new QLabel("Session", this);
    title->setStyleSheet("font-weight: 600; color: #d7dde6;");
    root->addWidget(title);

    dataset_value_ = addRow(*root, *this, "Dataset");
    data_range_value_ = addRow(*root, *this, "Data");
    visible_range_value_ = addRow(*root, *this, "Visible");
    layout_value_ = addRow(*root, *this, "Layout");
    replay_value_ = addRow(*root, *this, "Replay");
    message_value_ = addRow(*root, *this, "Message");
    root->addStretch();

    resetDataset();
}

void SideInfoPanelWidget::setDatasetName(const QString& name)
{
    dataset_value_->setText(name.isEmpty() ? "--" : name);
}

void SideInfoPanelWidget::setDataRange(const QString& range)
{
    data_range_value_->setText(range.isEmpty() ? "--" : range);
}

void SideInfoPanelWidget::setVisibleRange(const QString& range)
{
    visible_range_value_->setText(range.isEmpty() ? "--" : range);
}

void SideInfoPanelWidget::setLayoutSummary(const QString& summary)
{
    layout_value_->setText(summary.isEmpty() ? "--" : summary);
}

void SideInfoPanelWidget::setReplaySummary(const QString& summary)
{
    replay_value_->setText(summary.isEmpty() ? "--" : summary);
}

void SideInfoPanelWidget::setLastMessage(const QString& message)
{
    message_value_->setText(message.isEmpty() ? "--" : message);
}

void SideInfoPanelWidget::resetDataset()
{
    setDatasetName("No dataset loaded");
    setDataRange("--");
    setVisibleRange("--");
    setLayoutSummary("4 charts / Tabs");
    setReplaySummary("Disabled");
    setLastMessage("Ready");
}

} // namespace tradereview::app
