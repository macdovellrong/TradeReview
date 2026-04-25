#include "tradereview/data/DuckDbRepository.h"

#include <stdexcept>
#include <string>

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
#include <duckdb.h>
#endif

namespace tradereview::data {
namespace {

[[noreturn]] void throw_duckdb_unavailable()
{
    throw std::runtime_error(
        "DuckDB repository is unavailable because TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF");
}

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
[[noreturn]] void throw_duckdb_query_not_implemented(const char* operation)
{
    throw std::runtime_error(std::string{"DuckDB "} + operation + " is not implemented yet");
}
#endif

} // namespace

class DuckDbRepository::Impl {
public:
    Impl() = default;
    ~Impl()
    {
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        if (connection_ != nullptr) {
            duckdb_disconnect(&connection_);
        }
        if (database_ != nullptr) {
            duckdb_close(&database_);
        }
#endif
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    DataSetInfo open_readonly(const std::string& path)
    {
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        if (connection_ != nullptr) {
            duckdb_disconnect(&connection_);
        }
        if (database_ != nullptr) {
            duckdb_close(&database_);
        }

        const auto config_result = duckdb_create_config(&config_);
        if (config_result == DuckDBError) {
            throw std::runtime_error("failed to create DuckDB config");
        }
        const auto readonly_result = duckdb_set_config(config_, "access_mode", "READ_ONLY");
        if (readonly_result == DuckDBError) {
            duckdb_destroy_config(&config_);
            throw std::runtime_error("failed to set DuckDB read-only access mode");
        }
        const auto open_result = duckdb_open_ext(path.c_str(), &database_, config_, nullptr);
        duckdb_destroy_config(&config_);
        if (open_result == DuckDBError) {
            throw std::runtime_error("failed to open DuckDB database read-only");
        }
        if (duckdb_connect(database_, &connection_) == DuckDBError) {
            duckdb_close(&database_);
            throw std::runtime_error("failed to connect DuckDB database");
        }

        DataSetInfo info;
        info.dataset_path = path;
        info.metadata_only = true;
        return info;
#else
        (void)path;
        throw_duckdb_unavailable();
#endif
    }

private:
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    duckdb_database database_ = nullptr;
    duckdb_connection connection_ = nullptr;
    duckdb_config config_ = nullptr;
#endif
};

DuckDbRepository::DuckDbRepository()
    : impl_(std::make_unique<Impl>())
{
}

DuckDbRepository::~DuckDbRepository() = default;

DuckDbRepository::DuckDbRepository(DuckDbRepository&&) noexcept = default;

DuckDbRepository& DuckDbRepository::operator=(DuckDbRepository&&) noexcept = default;

DataSetInfo DuckDbRepository::open_readonly(const std::string& path)
{
    return impl_->open_readonly(path);
}

CandleWindow DuckDbRepository::query_candles(const CandleWindowRequest& request)
{
    (void)request;
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    throw_duckdb_query_not_implemented("candle query");
#else
    throw_duckdb_unavailable();
#endif
}

TickSlice DuckDbRepository::query_ticks(core::TimeRange range, size_t max_rows)
{
    (void)range;
    (void)max_rows;
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    throw_duckdb_query_not_implemented("tick query");
#else
    throw_duckdb_unavailable();
#endif
}

ReplayChunk DuckDbRepository::query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks)
{
    (void)from_ns;
    (void)to_ns;
    (void)max_ticks;
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    throw_duckdb_query_not_implemented("replay query");
#else
    throw_duckdb_unavailable();
#endif
}

} // namespace tradereview::data
