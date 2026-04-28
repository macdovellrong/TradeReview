#include "tradereview/chart/DrawingToolRailWidget.h"

#include <QPushButton>
#include <QString>
#include <QVBoxLayout>

#include <utility>

namespace tradereview::chart {
namespace {

constexpr int kToolRailWidth = 42;
constexpr int kRailButtonSize = 30;

QPushButton* addRailButton(QVBoxLayout& layout, QWidget& parent, const QString& label, const QString& action)
{
    auto* button = new QPushButton(label, &parent);
    button->setFixedSize(kRailButtonSize, kRailButtonSize);
    button->setToolTip(action);
    layout.addWidget(button);
    return button;
}

} // namespace

DrawingToolRailWidget::DrawingToolRailWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("DrawingToolRailWidget");
    setFixedWidth(kToolRailWidth);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 8, 6, 8);
    layout->setSpacing(6);

    const auto addButton = [this, layout](const QString& label, const QString& action) {
        auto* button = addRailButton(*layout, *this, label, action);
        connect(button, &QPushButton::clicked, this, [this, action](bool) {
            emitAction(action);
        });
    };

    addButton("S", "Sel");
    addButton("H", "H");
    addButton("V", "V");
    addButton("/", "Line");
    addButton("F", "Fib");
    addButton("E", "Fib Ext");
    addButton("C", "Clear");
    layout->addStretch();
}

void DrawingToolRailWidget::setDrawingActionCallback(DrawingActionCallback callback)
{
    drawing_action_callback_ = std::move(callback);
}

void DrawingToolRailWidget::emitAction(const QString& action) const
{
    if (drawing_action_callback_) {
        drawing_action_callback_(action);
    }
}

} // namespace tradereview::chart
