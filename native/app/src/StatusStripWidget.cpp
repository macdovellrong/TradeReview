#include "tradereview/app/StatusStripWidget.h"

#include "tradereview/app/AppTheme.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QString>

namespace tradereview::app {

StatusStripWidget::StatusStripWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("StatusStripWidget");
    setFixedHeight(theme::Size::StatusStripHeight);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(12, 0, 12, 0);
    layout->setSpacing(16);

    state_label_ = new QLabel("Ready", this);
    duckdb_label_ = new QLabel("DuckDB: build setting", this);
    data_range_label_ = new QLabel("Range: --", this);
    message_label_ = new QLabel("Native OpenGL chart ready", this);
    renderer_label_ = new QLabel("Renderer: OpenGL", this);

    layout->addWidget(state_label_);
    layout->addWidget(duckdb_label_);
    layout->addWidget(data_range_label_);
    layout->addWidget(message_label_, 1);
    layout->addWidget(renderer_label_);
}

void StatusStripWidget::setStateText(const QString& text)
{
    state_label_->setText(text);
}

void StatusStripWidget::setDuckDbText(const QString& text)
{
    duckdb_label_->setText(text);
}

void StatusStripWidget::setDataRangeText(const QString& text)
{
    data_range_label_->setText(text);
}

void StatusStripWidget::setRendererText(const QString& text)
{
    renderer_label_->setText(text);
}

void StatusStripWidget::setMessageText(const QString& text)
{
    message_label_->setText(text);
}

} // namespace tradereview::app
