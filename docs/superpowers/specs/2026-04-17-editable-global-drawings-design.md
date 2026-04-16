# Editable Global Drawings Design

Date: 2026-04-17
Project: TradeReview
Scope: Add TradingView-style drawing selection and direct manipulation, add a rectangle drawing tool, and persist drawings globally across app restarts using time and price coordinates.

## 1. Context

The current drawing implementation in `ui/main_window.py` supports creating and rendering:

- `hline`
- `vline`
- `line`
- `fib`
- `fib_ext`

The recent refactor moved Fibonacci math, drawing spec normalization, and multi-point drawing sessions into `ui/drawings/`, but the runtime behavior is still limited in three important ways:

- existing drawings are not directly editable after creation
- there is no rectangle drawing tool
- drawings only exist in memory and are lost after restart

The user wants the interaction model to move closer to software such as TradingView:

- click a drawing to select it
- show visible handles when selected
- drag handles to edit the drawing
- drag the drawing body to move it
- show the same drawings on every chart period
- persist drawings globally, independent of the currently loaded data file

The user explicitly does not want drawings tied to a specific database file. Drawings should be anchored by `time + price` coordinates and restored globally on next launch.

## 2. Goals

- Add direct selection and manipulation for supported drawing types.
- Add a new `Rect` toolbar tool for rectangle creation and editing.
- Preserve global cross-chart synchronization so the same drawing appears on every enabled chart.
- Persist all drawings globally using `QSettings`.
- Keep drawing coordinates independent from the loaded market data file.
- Use an architecture that separates chart rendering concerns from drawing interaction concerns.
- Introduce dedicated editable drawing item classes so future tools can follow the same pattern.

## 3. Non-Goals

- No property panel for manual numeric editing in this iteration.
- No snapping to candles, highs/lows, or other guides in this iteration.
- No rotation handles, copy/paste, locking, hiding, or z-order controls in this iteration.
- No drawing folders, templates, or per-symbol/per-file drawing libraries in this iteration.
- No attempt to perfectly match TradingView interaction nuance in the first pass.

## 4. Confirmed Product Decisions

- Use a `3 + 2` architecture:
  - a dedicated drawing subsystem separate from the chart core
  - specialized editable drawing item classes per tool
- Drawings are global, not associated with the currently loaded DuckDB or parquet file.
- Drawing anchors are stored as time and price, not candle indices and not screen pixels.
- The same drawings should appear on all chart periods.
- Editing a drawing on any chart updates the same global drawing and propagates to all charts.
- The initial interaction scope includes:
  - selection
  - visible handles
  - drag-to-edit
  - drag-to-move
  - rectangle creation
  - automatic persistence

## 5. Architecture

Introduce a dedicated drawing subsystem under `ui/drawings/` with the following responsibilities.

### 5.1 GlobalDrawingStore

Responsible for the canonical set of normalized drawing specs.

Responsibilities:

- allocate stable drawing ids
- store drawings keyed by id
- add, update, delete, and clear drawings
- expose ordered iteration for rendering and hit-test precedence
- serialize and deserialize drawings for persistence

This store is the single source of truth. Chart-local pyqtgraph items are derived views and must not become the canonical state.

### 5.2 DrawingController

Responsible for interaction state and mutation orchestration.

Responsibilities:

- track the selected drawing id
- start drawing sessions for creation tools
- resolve whether a mouse press targets a handle, a drawing body, or empty space
- manage drag lifecycle
- commit drawing updates back to the store
- notify persistence after create, update, delete, or clear operations

The controller operates on normalized specs and delegates chart-specific display to chart-local adapters.

### 5.3 EditableDrawingItem Base Contract

Each editable drawing class should represent one drawing instance on one chart and implement a shared contract.

Required capabilities:

- apply a normalized drawing spec
- render normal and selected states
- expose selectable handles
- determine whether a screen position hits a handle or the drawing body
- update an in-memory preview spec during drag
- remove all underlying pyqtgraph items from the plot

This contract allows the controller and chart adapter to work with different tools through one interface.

### 5.4 Tool-Specific Editable Items

The first wave of dedicated editable item classes will cover:

- `EditableHLineItem`
- `EditableVLineItem`
- `EditableLineItem`
- `EditableFibItem`
- `EditableFibExtensionItem`
- `EditableRectItem`

Each class owns:

- body item creation
- selected styling
- handle item creation and layout
- tool-specific drag behavior

### 5.5 ChartDrawingLayer

Each `ChartWidget` gets a drawing layer adapter that bridges the global store to the local chart.

Responsibilities:

- create editable item instances for drawings visible on that chart
- keep a local map of `drawing_id -> EditableDrawingItem`
- update or replace only the affected drawing item when a store entry changes
- keep local selected state aligned with the controller
- convert plot coordinates to `time + price` for interaction

The chart widget continues to own market-data rendering, crosshair behavior, replay, zoom, and pan.

### 5.6 Persistence

Drawings are persisted independently from view state.

Responsibilities:

- save global drawings to `QSettings`
- restore drawings during application startup
- tolerate malformed stored data and recover with an empty drawing set

`Save View` continues to persist view navigation state only. Drawing persistence is automatic and not gated by a button.

## 6. UI and Interaction Design

### 6.1 Toolbar

Keep the existing drawing toolbar and add `Rect`.

Expected toolbar set:

- `Sel`
- `H`
- `V`
- `Line`
- `Fib`
- `Fib Ext`
- `Rect`
- `Fib Config`
- `Clear`

`Sel` becomes the explicit drawing-edit mode. Creation tools still work as mode-based tools.

### 6.2 Selection Model

In `Sel` mode:

- left-click on a drawing selects it
- selected drawing shows highlight styling and visible handles
- left-click on empty space clears selection
- pressing `Delete` removes the selected drawing
- right-click `Delete Drawing` continues to work on the selected drawing

When a creation tool is active, click behavior is reserved for creating that tool's points and does not enter edit mode.

### 6.3 Drag Lifecycle

For editable drawings:

1. Mouse press determines whether the target is a handle, the drawing body, or empty space.
2. Mouse move updates a preview spec in memory and refreshes the local drawing item in place.
3. Mouse release commits the updated spec to `GlobalDrawingStore`.
4. The store update propagates to all charts and triggers persistence.

Persistence happens on commit, not on every mouse move.

### 6.4 Tool-Specific Interaction Rules

#### HLine

- body selection targets the horizontal line
- dragging the body changes the stored price
- no horizontal movement effect is needed because the line spans the chart

#### VLine

- body selection targets the vertical line
- dragging the body changes the stored timestamp
- no vertical movement effect is needed because the line spans the chart

#### Line

- two endpoint handles are visible when selected
- dragging an endpoint moves only that endpoint
- dragging the line body translates both endpoints by the same time and price delta

#### Fib

- two anchor handles are visible when selected
- dragging an anchor updates that anchor and recomputes all retracement levels
- dragging the body translates both anchors by the same time and price delta
- level labels update from the moved anchors using the snapshot levels stored on the drawing

#### Fib Ext

- three anchor handles are visible when selected
- dragging any anchor recomputes extension levels
- dragging the body translates all three anchors by the same time and price delta
- level labels update from the moved anchors using the snapshot levels stored on the drawing

#### Rect

- creation uses two diagonal points
- selected rectangles show four corner handles
- dragging a corner updates rectangle bounds using the opposite corner as the fixed reference
- dragging the body translates the entire rectangle by the same time and price delta

### 6.5 Clear Behavior

`Clear` clears the global drawing store, not just the current chart.

All charts should remove all drawing items immediately and persistence should write the empty state.

## 7. Data Model

Continue using normalized drawing specs and extend them to cover rectangles and persisted storage.

Example line spec:

```python
{
    "id": 8,
    "type": "line",
    "points": [
        {"dt": ..., "price": 3300.0},
        {"dt": ..., "price": 3320.0},
    ],
}
```

Example rectangle spec:

```python
{
    "id": 17,
    "type": "rect",
    "points": [
        {"dt": ..., "price": 3301.25},
        {"dt": ..., "price": 3320.50},
    ],
}
```

Example Fibonacci extension spec:

```python
{
    "id": 22,
    "type": "fib_ext",
    "points": [
        {"dt": ..., "price": 3300.0},
        {"dt": ..., "price": 3330.0},
        {"dt": ..., "price": 3315.0},
    ],
    "config_snapshot": {
        "levels": [0.618, 1.0, 1.272, 1.618, 2.0],
    },
}
```

Supported persisted types in this iteration:

- `hline`
- `vline`
- `line`
- `fib`
- `fib_ext`
- `rect`

## 8. Persistence Format

Persist drawings through the existing application `QSettings` instance:

```python
QSettings("TradeReview", "TradeReview")
```

Use dedicated drawing keys, for example:

- `drawings/version`
- `drawings/items`

`drawings/items` stores a JSON array of normalized drawing specs.

Persisted JSON example:

```json
[
  {
    "id": 17,
    "type": "rect",
    "points": [
      {"dt": "2026-04-17T09:30:00-04:00", "price": 3301.25},
      {"dt": "2026-04-17T11:00:00-04:00", "price": 3320.5}
    ]
  }
]
```

Persistence rules:

- timestamps are stored as ISO8601 strings with timezone information when available
- prices are stored as floats
- new drawings are saved automatically after creation
- edited drawings are saved automatically after drag commit
- deleted or cleared drawings are saved immediately
- selection state is not persisted

Error handling rules:

- malformed JSON falls back to an empty drawing list
- malformed individual records are skipped instead of aborting the entire restore
- unknown drawing types are ignored

## 9. Coordinate Mapping

The user wants drawings tied to `time + price`, not to a specific file's candle indices. This requires a more flexible time-to-x mapping than the current strict clamp behavior.

Requirements:

- if a drawing timestamp matches an in-range data point, use the aligned x index
- if a drawing timestamp is before the first visible timestamp, extrapolate left using the chart time delta
- if a drawing timestamp is after the last visible timestamp, extrapolate right using the chart time delta
- do not silently collapse out-of-range drawings to the first or last candle

This preserves the intent of a global drawing library even when the currently loaded data does not fully cover the drawing's timestamps.

## 10. Rendering and Hit Testing

### 10.1 Rendering Strategy

Store changes should update only affected drawing instances on each chart.

- add: create one new editable item
- update: replace or refresh the corresponding editable item
- delete: remove only that editable item
- clear: remove all editable items

Avoid full `clear_drawings()` plus full rebuild on every single update because it will increase flicker and make future interaction work harder to reason about.

### 10.2 Hit Testing

Hit testing should prioritize:

1. handles
2. drawing body
3. empty space

Hit tolerance should be based on screen-space distance rather than only raw data-space distance, because zoom level changes otherwise make selection inconsistent.

For overlapping drawings, prefer:

1. most recently selected drawing
2. otherwise most recently created drawing

This gives a stable and user-visible selection rule.

### 10.3 Selected Styling

Selected drawings should visibly differ from unselected drawings through:

- thicker or brighter body lines
- visible handle markers
- persistent selected state until the user selects another drawing or clicks empty space

Unselected drawings do not show handles.

## 11. Testing Strategy

Add focused coverage for the new architecture and behaviors.

### 11.1 Store and Persistence Tests

Create tests for:

- add, update, delete, and clear operations
- id allocation
- JSON serialization and deserialization
- malformed persistence payload recovery

### 11.2 Spec and Tool Tests

Extend or add tests for:

- rectangle spec normalization
- persisted timestamp parsing
- legacy spec compatibility for existing tools
- rectangle creation session behavior

### 11.3 Editable Item Tests

Add tests for:

- line endpoint dragging updates the correct anchor
- line body dragging translates both anchors
- rectangle corner dragging updates bounds correctly
- Fibonacci anchor dragging recomputes derived levels
- selected state shows handles only when active

### 11.4 Chart Integration Tests

Add tests for:

- selection in `Sel` mode
- editing on one chart propagates to another chart
- rectangle creation through the chart widget
- deletion of the selected drawing
- clear removes drawings across all charts

### 11.5 Session and Startup Tests

Keep existing view-state tests and extend them to verify:

- drawings restore at startup
- drawings persist independently from the current database file
- broken persisted drawing payloads do not block application startup

## 12. Implementation Notes and Risks

- The first pass should keep controller-led event routing. Editable items should expose hit-testing and update methods, but the controller remains responsible for deciding the active interaction.
- `Fib` and `Fib Ext` body-drag behavior may initially be implemented as uniform anchor translation without advanced snapping or geometry constraints.
- pyqtgraph item classes may need thin wrappers or composition rather than deep subclassing if event behavior becomes unstable.
- Keyboard deletion should be added carefully so it only applies when a drawing is selected and the chart widget has focus.

## 13. Outcome

After this work:

- drawings can be selected and directly edited
- rectangles can be created and edited
- drawings survive application restarts
- all periods show the same global drawing set
- the codebase has a dedicated drawing subsystem that can support future tools such as channels, rays, and text annotations
