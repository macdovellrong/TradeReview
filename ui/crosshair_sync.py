class CrosshairSyncController:
    def __init__(self):
        self._charts = []

    def register_chart(self, chart):
        if chart not in self._charts:
            self._charts.append(chart)

    def unregister_chart(self, chart):
        if chart in self._charts:
            self._charts.remove(chart)

    def sync_from(self, source_chart, timestamp, price):
        for chart in list(self._charts):
            if chart is source_chart:
                continue
            chart.sync_crosshair(timestamp, price)
