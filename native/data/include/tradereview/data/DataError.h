#pragma once

#include <stdexcept>
#include <string>
#include <utility>

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

class DataException : public std::runtime_error {
public:
    explicit DataException(DataError error)
        : std::runtime_error(error.message.empty() ? "data error" : error.message)
        , error_(std::move(error))
    {
    }

    [[nodiscard]] const DataError& error() const noexcept
    {
        return error_;
    }

private:
    DataError error_;
};

} // namespace tradereview::data
