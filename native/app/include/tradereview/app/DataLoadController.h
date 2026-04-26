#pragma once

#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataSetInfo.h"

#include <QString>

#include <memory>

namespace tradereview::chart {
class ChartWorkspaceWidget;
}

namespace tradereview::data {
class IDataStore;
}

namespace tradereview::app {

struct LoadResult {
    data::DataSetInfo dataset_info;
    data::CandleWindow window;
};

class DataLoadController final {
public:
    DataLoadController();
    explicit DataLoadController(std::unique_ptr<data::IDataStore> store);
    ~DataLoadController();

    DataLoadController(const DataLoadController&) = delete;
    DataLoadController& operator=(const DataLoadController&) = delete;
    DataLoadController(DataLoadController&&) noexcept;
    DataLoadController& operator=(DataLoadController&&) noexcept;

    LoadResult load_file(const QString& path, chart::ChartWorkspaceWidget& workspace);

private:
    std::unique_ptr<data::IDataStore> store_;
};

} // namespace tradereview::app
