#pragma once

#include <string>

namespace tradereview::data {

enum class DataErrorCode {
    None = 0,
    FileNotFound,
    OpenFailed,
    SchemaMismatch,
    MissingTable,
    MissingColumn,
    QueryFailed,
    InvalidRequest,
};

struct DataError {
    DataErrorCode code = DataErrorCode::None;
    std::string message;
    std::string path;
    std::string table;
};

} // namespace tradereview::data
