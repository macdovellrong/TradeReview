from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ui.drawings.fib_config import (
    DEFAULT_EXTENSION_PRESETS,
    DEFAULT_RETRACEMENT_PRESETS,
    FibLevelsConfig,
    FibSettings,
    default_fib_settings,
    merge_fib_levels,
)


class FibConfigDialog(QDialog):
    def __init__(self, fib_settings: FibSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fib Config")
        self.retracement_checkboxes: dict[float, QCheckBox] = {}
        self.extension_checkboxes: dict[float, QCheckBox] = {}
        self.retracement_custom_edit = QLineEdit(fib_settings.retracement.custom_levels_text)
        self.extension_custom_edit = QLineEdit(fib_settings.extension.custom_levels_text)

        layout = QVBoxLayout(self)
        layout.addWidget(
            self._build_group(
                "Retracement Levels",
                DEFAULT_RETRACEMENT_PRESETS,
                fib_settings.retracement.enabled_levels,
                self.retracement_checkboxes,
                self.retracement_custom_edit,
            )
        )
        layout.addWidget(
            self._build_group(
                "Extension Levels",
                DEFAULT_EXTENSION_PRESETS,
                fib_settings.extension.enabled_levels,
                self.extension_checkboxes,
                self.extension_custom_edit,
            )
        )

        button_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.on_reset_defaults)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        button_row.addWidget(buttons)
        layout.addLayout(button_row)

    def _build_group(self, title, presets, enabled_levels, checkbox_map, line_edit):
        group = QGroupBox(title)
        form = QFormLayout(group)
        enabled = set(float(level) for level in enabled_levels)
        for level in presets:
            checkbox = QCheckBox(f"{level:g}")
            checkbox.setChecked(level in enabled)
            checkbox_map[level] = checkbox
            form.addRow(checkbox)
        form.addRow("Custom", line_edit)
        return group

    def _selected_levels(self, checkbox_map: dict[float, QCheckBox]) -> list[float]:
        return [level for level, checkbox in checkbox_map.items() if checkbox.isChecked()]

    def build_settings(self) -> FibSettings:
        retracement_enabled = self._selected_levels(self.retracement_checkboxes)
        extension_enabled = self._selected_levels(self.extension_checkboxes)
        merge_fib_levels(retracement_enabled, self.retracement_custom_edit.text())
        merge_fib_levels(extension_enabled, self.extension_custom_edit.text())
        return FibSettings(
            retracement=FibLevelsConfig(
                enabled_levels=retracement_enabled,
                custom_levels_text=self.retracement_custom_edit.text(),
            ),
            extension=FibLevelsConfig(
                enabled_levels=extension_enabled,
                custom_levels_text=self.extension_custom_edit.text(),
            ),
        )

    def on_reset_defaults(self):
        defaults = default_fib_settings()
        for level, checkbox in self.retracement_checkboxes.items():
            checkbox.setChecked(level in defaults.retracement.enabled_levels)
        for level, checkbox in self.extension_checkboxes.items():
            checkbox.setChecked(level in defaults.extension.enabled_levels)
        self.retracement_custom_edit.setText(defaults.retracement.custom_levels_text)
        self.extension_custom_edit.setText(defaults.extension.custom_levels_text)

    def on_accept(self):
        try:
            self.build_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Fib Config", str(exc))
            return
        self.accept()
