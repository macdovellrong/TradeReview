from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadResult:
    success: bool
    file_path: str
    initial_time: object = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class DataLoadingFacade:
    def __init__(self, engine):
        self.engine = engine

    def load(self, file_path: str) -> DataLoadResult:
        self.engine.parquet_file = file_path
        self.engine.load_data()
        df_ticks = self.engine.df_ticks
        if df_ticks is None or df_ticks.empty:
            return DataLoadResult(
                success=False,
                file_path=file_path,
                error=self.engine.last_load_error or "Failed to load the selected data file.",
            )

        total_ticks = len(df_ticks)
        if total_ticks > 100000:
            initial_time = df_ticks.index[100000]
        else:
            initial_time = df_ticks.index[0]

        return DataLoadResult(
            success=True,
            file_path=file_path,
            initial_time=initial_time,
            warnings=tuple(self.engine.last_load_warnings or ()),
        )
