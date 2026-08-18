# Aerospace Codebase Harness Research

## 1. Minable Open-Source Aerospace Repos

| Repo | Language | What It Does | Mineable? |
|---|---|---|---|
| [nyx-space/nyx](https://github.com/nyx-space/nyx) | **Rust** | High-fidelity astrodynamics (orbit propagation, lunar missions — CAPSTONE, Blue Ghost 1) | Yes, already supported |
| [poliastro/poliastro](https://github.com/poliastro/poliastro) | **Python** | Astrodynamics library (orbit propagation, Lambert problem) | Yes, already supported |
| [nasa/fprime](https://github.com/nasa/fprime) | **C++/Python** | Flight software framework for CubeSats/SmallSats | Needs C++ config |
| [nasa/cfs](https://github.com/nasa/cfs) | **C** | Core Flight System — used on flagship NASA missions, 12+ sub-apps | Needs C config |
| [JSBSim-Team/jsbsim](https://github.com/JSBSim-Team/jsbsim) | **C++** | Flight dynamics model library, Python bindings | Needs C++ config |
| [ericstoneking/42](https://github.com/ericstoneking/42) | **C** | NASA spacecraft attitude control simulation | Needs C config |
| [nasa/openmct](https://github.com/nasa/openmct) | **JS/TS** | Web-based mission control framework | Yes, already supported (nodejs) |
| [gnss-sdr/gnss-sdr](https://github.com/gnss-sdr/gnss-sdr) | **C++** | Open-source GNSS software-defined receiver | Needs C++ config |

## 2. Domain-Specific Test Frameworks

- **SITL (Software-In-The-Loop)**: Runs flight code on desktop without hardware. F Prime and cFS support this natively. Fits in Docker.
- **F Prime GDS**: Python-based integration test harness — sends commands, asserts on telemetry
- **cFS UT Assert**: Built-in C-based unit test framework with coverage reporting
- **JSBSim scripting**: XML-based test scenarios that define maneuvers and assert on flight state
- **HIL (Hardware-In-The-Loop)**: Requires physical boards — **out of scope** for Docker pipelines

## 3. Build Toolchains

- **CMake** dominates C/C++ aerospace (JSBSim, GNSS-SDR, cFS, 42, F Prime)
- **F Prime** uses custom CMake + Python: `fprime-util generate && fprime-util build`
- **Cross-compilation** (ARM CubeSats, RTEMS/VxWorks) — **out of scope**, stick to native x86 in Docker
- **Cargo** for Rust (nyx-space) — already supported
- **Meson** used by some GNSS projects

## 4. Verification/Validation Tools

| Tool | Type | Open Source? | Useful for us? |
|---|---|---|---|
| Frama-C | Formal verification for C | Yes | Possible future integration |
| cppcheck + MISRA addon | Static analysis | Yes | Possible setup_command |
| clang-tidy | Static analysis | Yes | Possible setup_command |
| gcov/lcov | MC/DC coverage (DO-178C) | Yes | Test coverage metrics |
| SPIN/ESBMC | Model checking | Yes | Research-grade, niche |
| DO-178C compliance | Process/documentation | N/A | Out of scope (not a tool) |

## 5. LangConfig Additions Needed

### C/C++ CMake-based projects (cFS, JSBSim, GNSS-SDR, 42)

```python
CPP = LangConfig(
    name="cpp",
    docker_image="gcc:13-bookworm",
    build_cmd="mkdir -p build && cd build && cmake .. && make -j$(nproc)",
    test_cmd="cd build && ctest --output-on-failure",
    file_extensions=(".c", ".cpp", ".h", ".hpp", ".cmake"),
    useful_commands=(...),
    setup_commands=(
        "apt-get update -qq && apt-get install -y -qq cmake make git > /dev/null 2>&1",
    ),
)
```

### F Prime (specialized C++ + Python flight software)

```python
FPRIME = LangConfig(
    name="fprime",
    docker_image="python:3.11-slim",
    build_cmd="fprime-util generate && fprime-util build",
    test_cmd="fprime-util check",
    file_extensions=(".cpp", ".hpp", ".fpp", ".py"),
    useful_commands=(...),
    setup_commands=(
        "pip install fprime-tools fprime-gds",
        "apt-get update -qq && apt-get install -y -qq cmake make g++ > /dev/null 2>&1",
    ),
)
```

### Task Miner Updates Needed

- Add `cpp` test file patterns: `*_test.cpp`, `*Test.cpp`, `test_*.cpp`, `*_tests.cpp`
- Add C/C++ test name extraction: `TEST(suite, name)` (Google Test), `TEST_CASE(name)` (Catch2)

## 6. In Scope vs Out of Scope

| In Scope | Out of Scope |
|---|---|
| Rust repos (nyx-space) — works today | HIL testing (needs hardware) |
| Python repos (poliastro) — works today | RTOS cross-compilation (VxWorks/RTEMS) |
| JS/TS repos (OpenMCT) — works today | Commercial MISRA/DO-178C tools |
| C/C++ CMake repos — needs new config | Java-based Orekit (needs JVM config) |
| F Prime — needs specialized config | Real-time embedded targets |

## Sources

- [NASA F Prime](https://github.com/nasa/fprime)
- [NASA cFS](https://github.com/nasa/cfs)
- [poliastro](https://github.com/poliastro/poliastro)
- [nyx-space](https://github.com/nyx-space/nyx)
- [JSBSim](https://github.com/JSBSim-Team/jsbsim)
- [42 Spacecraft Sim](https://github.com/ericstoneking/42)
- [GNSS-SDR](https://github.com/gnss-sdr/gnss-sdr)
- [NASA OpenMCT](https://github.com/nasa/openmct)
- [AeroRust awesome-space](https://github.com/AeroRust/awesome-space)
- [Orbital Index awesome-space](https://github.com/orbitalindex/awesome-space)
