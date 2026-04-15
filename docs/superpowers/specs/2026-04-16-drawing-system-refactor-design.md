# Drawing System Refactor Design

Date: 2026-04-16
Project: TradeReview
Scope: Refactor the chart drawing system to support configurable Fibonacci retracement levels, add a standard three-point Fibonacci extension tool, and establish an extensible drawing architecture for future tools such as rectangles and channels.

## 1. Context

The current drawing implementation lives mostly inside `ui/main_window.py` and mixes together:

- toolbar button wiring
- active drawing mode state
- mouse click and mouse move handling
- drawing spec creation
- pyqtgraph item rendering
- selection and deletion behavior
- cross-chart drawing synchronization

This structure was acceptable for the initial set of tools (`hline`, `vline`, `line`, `fib`), but it is not a good base for adding more multi-point tools. The immediate feature request is:

- make Fibonacci retracement levels configurable
- include retracement levels such as `0.5`, `0.618`, `0.7`, `0.786`, `0.8`
- add a configurable Fibonacci extension tool
- use a standard three-point `A-B-C` extension drawing workflow
- persist Fibonacci configuration globally so it only needs to be set once
- do not retroactively redraw existing Fibonacci objects after the config changes

The user also expects more drawing tools later, such as rectangles and channels, so this work should improve the system architecture instead of adding more type-specific conditionals.

## 2. Goals

- Preserve existing line, horizontal line, vertical line, selection, deletion, clear, and cross-chart synchronization behavior.
- Keep the existing `Fib` tool name and user-facing label.
- Add a new `Fib Ext` tool that uses a standard three-point `A-B-C` workflow.
- Add a `Fib Config` toolbar button with a global settings dialog.
- Support both preset checkbox selection and free-form custom ratio input for retracement and extension levels.
- Persist Fibonacci configuration in `QSettings`.
- Snapshot Fibonacci levels into each created drawing so later config changes only affect new drawings.
- Refactor the drawing system so future tools can declare their own point count, preview behavior, and renderer with minimal changes to existing code.

## 3. Non-Goals

- No drag editing or resizing of existing drawings in this iteration.
- No live restyling or re-leveling of existing Fibonacci objects after config changes.
- No implementation of rectangles, channels, rays, or other new tools in this iteration.
- No change to the existing multi-chart broadcast model for created drawings.
- No migration of existing persisted drawing data because drawings are not currently persisted across restarts.

## 4. Confirmed Product Decisions

- `Fib` remains named `Fib`; no rename is required.
- Fibonacci extension uses the standard three-point `A-B-C` method.
- Fibonacci configuration is global and persistent.
- The configuration UI is a dedicated toolbar button (`Fib Config`).
- The config dialog supports both preset checkboxes and editable custom ratio input.
- Default extension levels are `0.618`, `1.0`, `1.272`, `1.618`, `2.0`.
- Existing drawings do not update when Fibonacci config changes.

## 5. Proposed Architecture

Introduce a small drawing subsystem under `ui/` with four responsibilities:

### 5.1 Drawing Config

Responsible for:

- default Fibonacci preset levels
- parsing custom level input
- merging preset and custom levels
- validation, deduplication, and sorting
- saving and loading drawing settings from `QSettings`

This layer should be pure logic plus a thin persistence wrapper so it can be tested without Qt widgets beyond `QSettings`.

### 5.2 Drawing Tool State

Responsible for:

- active tool selection
- how many points the active tool requires
- accumulating clicked points
- building preview state from the current cursor position
- producing a normalized drawing spec when enough points have been collected

This removes the need for scattered conditionals like `if self.draw_mode in ("line", "fib")`.

### 5.3 Drawing Registry

Responsible for registering drawing tools with metadata such as:

- tool id
- toolbar label
- required point count
- renderer key
- whether the tool depends on Fibonacci config

The registry gives future tools a clear integration point.

### 5.4 Drawing Renderer

Responsible for converting a normalized drawing spec into one or more pyqtgraph items and tagging those items with drawing metadata for selection and deletion.

Renderers will initially exist for:

- `hline`
- `vline`
- `line`
- `fib`
- `fib_ext`

## 6. UI and Interaction Design

### 6.1 Toolbar

The chart toolbar will contain:

- `Sel`
- `H`
- `V`
- `Line`
- `Fib`
- `Fib Ext`
- `Fib Config`
- `Clear`

`Sel` keeps the current behavior of leaving drawing mode.

### 6.2 Fibonacci Retracement Workflow

`Fib` remains a two-point tool:

1. First click records point 1.
2. Mouse movement previews the span.
3. Second click records point 2 and creates the drawing.

Rendered output:

- top and bottom boundary lines at the two anchor prices
- one horizontal level line per enabled retracement ratio
- a right-side label for each level showing ratio and price

### 6.3 Fibonacci Extension Workflow

`Fib Ext` is a three-point tool using standard `A-B-C` behavior:

1. First click records `A`.
2. Second click records `B`.
3. Third click records `C` and creates the extension drawing.

Preview behavior:

- after `A`, moving the mouse previews the `A-B` segment
- after `A-B`, moving the mouse previews the `A-B-C` structure and extension projection

Rendered output:

- optional faint dashed guide segments for `A-B` and `B-C`
- one horizontal level line per enabled extension ratio
- a right-side label for each level showing ratio and price

Extension price calculation:

- measure the segment `B - A`
- project from `C`
- price formula: `C + (B - A) * level`

This automatically handles both upward and downward structures because the sign of `B - A` is preserved.

### 6.4 Fib Config Dialog

The `Fib Config` dialog is a global settings editor with two sections:

- `Retracement Levels`
- `Extension Levels`

Each section includes:

- preset ratio checkboxes
- a custom comma-separated input field
- an effective levels preview or summary

Dialog buttons:

- `Save`
- `Cancel`
- `Reset Defaults`

Behavior:

- `Save` validates and persists the config
- `Cancel` closes without applying changes
- `Reset Defaults` restores built-in defaults in the dialog before saving

## 7. Data Model

All tools should move to a normalized drawing spec that stores points in a list instead of tool-specific `p1` and `p2` fields.

Example retracement spec:

```python
{
    "id": 12,
    "type": "fib",
    "points": [
        {"dt": ..., "price": ...},
        {"dt": ..., "price": ...},
    ],
    "config_snapshot": {
        "levels": [0.5, 0.618, 0.7, 0.786, 0.8],
    },
}
```

Example extension spec:

```python
{
    "id": 13,
    "type": "fib_ext",
    "points": [
        {"dt": ..., "price": ...},
        {"dt": ..., "price": ...},
        {"dt": ..., "price": ...},
    ],
    "config_snapshot": {
        "levels": [0.618, 1.0, 1.272, 1.618, 2.0],
    },
}
```

Compatibility rule:

- during migration, drawing creation and rendering should accept legacy specs using `p1_dt`, `p1_price`, `p2_dt`, `p2_price`
- all new drawing creation should emit normalized `points`

Why `config_snapshot` is required:

- existing drawings must not change when global config changes
- renderer behavior must remain deterministic for each saved spec

## 8. Fibonacci Configuration Model

The config layer stores separate settings for retracement and extension:

- selected preset levels
- custom level string or parsed custom values
- final effective levels

Default retracement presets should include at least:

- `0.236`
- `0.382`
- `0.5`
- `0.618`
- `0.7`
- `0.786`
- `0.8`

Default extension presets should include:

- `0.618`
- `1.0`
- `1.272`
- `1.618`
- `2.0`

Parsing and normalization rules:

- accept comma-separated numeric input
- trim whitespace
- ignore empty tokens
- reject non-numeric tokens
- reject negative values
- deduplicate repeated values
- sort ascending

The dialog should surface validation errors instead of silently accepting invalid custom input.

## 9. Rendering Model

Each renderer returns the pyqtgraph items created for one drawing. A lightweight wrapper such as `RenderedDrawing(items=[...])` is acceptable if it simplifies future selection or styling logic.

Renderer responsibilities:

- convert anchor datetimes to chart x positions
- create the pyqtgraph items
- assign `_is_drawing = True`
- assign `_drawing_id = <id>`
- return all created items so the chart can remove them later

Fib retracement renderer:

- draw boundary lines at anchor prices across the anchor x span
- compute each level from the two anchor prices
- draw dashed horizontal lines for configured levels
- draw level labels on the right side

Fib extension renderer:

- compute extension levels from `A-B-C`
- draw faint guide lines for structure if included
- draw dashed horizontal lines for configured levels
- draw level labels on the right side

## 10. Migration Strategy

Refactor in controlled steps:

1. Extract Fibonacci config and math into testable modules.
2. Add normalized spec creation and compatibility helpers.
3. Introduce a drawing tool registry and point-collection state model.
4. Move rendering logic out of `ChartWidget.add_drawing`.
5. Wire the new toolbar button and config dialog.
6. Keep existing selection, delete, clear, and cross-chart broadcast behavior intact.

This staged approach reduces regression risk while still delivering the structural improvement needed for future tools.

## 11. Testing Strategy

Follow TDD for implementation. Focus automated tests on pure logic first.

Planned tests:

- Fibonacci config defaults
- preset and custom level merging
- invalid custom level handling
- `QSettings` save/load round trip
- retracement price calculation
- extension `A-B-C` price calculation for upward and downward cases
- drawing tool point-count and completion behavior
- legacy spec compatibility conversion

Optional widget-level tests may be added if they remain lightweight, but the primary regression safety should come from pure logic tests.

## 12. Risks and Mitigations

- Risk: regressions in existing line tools
  - Mitigation: keep renderer scope isolated and preserve current broadcast/delete flow
- Risk: over-scoping into a full drawing editor
  - Mitigation: explicitly defer drag editing and additional tools
- Risk: config parsing ambiguity
  - Mitigation: validate strictly and display errors in the config dialog
- Risk: refactor churn in `ui/main_window.py`
  - Mitigation: move logic outward in small steps and keep compatibility adapters during migration

## 13. Implementation Boundaries for This Iteration

Included:

- drawing system refactor sufficient for normalized point-based tools
- configurable Fibonacci retracement
- standard three-point Fibonacci extension
- global persistent Fibonacci configuration dialog

Excluded:

- draggable editing handles
- object restyling after creation
- rectangle and channel implementation
- persistent storage of drawings across app restarts

## 14. Acceptance Criteria

- `Fib` supports configurable retracement ratios and includes `0.5`, `0.618`, `0.7`, `0.786`, `0.8` by default.
- `Fib Ext` exists and uses three-point `A-B-C` extension logic.
- `Fib Config` allows preset selection and custom ratio input for both retracement and extension.
- Fibonacci config is stored globally and restored on app restart.
- Existing Fibonacci drawings do not change after config updates.
- Existing line, selection, deletion, clear, and multi-chart synchronization behaviors continue to work.
- The codebase has automated tests covering Fibonacci config parsing/persistence and core Fibonacci math.
