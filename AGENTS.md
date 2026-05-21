# AGENTS

## Repo layout
- `tasks/task1-aruco` (Python), `tasks/task2-detector` (Python), `tasks/task3-tracker` (C++); root CMake only builds Task 3.

## Setup (Python)
- Use uv, not pip: `uv sync --extra vision --extra train --extra dev`.

## Tests
- Python: `uv run pytest` (auto-builds the Task 3 C++ target when needed).
- Optional Task 2 training tests: `uv run pytest --run-task2-training`.
- C++ (macOS/Linux): `cmake -S . -B build/hw -DHW_BUILD_TESTS=ON`, `cmake --build build/hw`, `ctest --test-dir build/hw --output-on-failure`.
- C++ (Windows): `cmake -S . -B build/hw-ninja -G Ninja -DHW_BUILD_TESTS=ON`, `cmake --build build/hw-ninja`, `ctest --test-dir build/hw-ninja --output-on-failure`.

## Task data + demos
- Task 1 input data must live in `tasks/task1-aruco/data/calibration/` and `tasks/task1-aruco/data/aruco/`.
- Simulator demo: launch the platform binary under `simulator/`, then run `uv run python simulator/runner.py` (optional `--seed <n>`).
