#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::chart {

class DrawingToolRailWidget final : public QWidget {
public:
    using DrawingActionCallback = std::function<void(const QString&)>;

    explicit DrawingToolRailWidget(QWidget* parent = nullptr);

    void setDrawingActionCallback(DrawingActionCallback callback);

private:
    void emitAction(const QString& action) const;

    DrawingActionCallback drawing_action_callback_;
};

} // namespace tradereview::chart
