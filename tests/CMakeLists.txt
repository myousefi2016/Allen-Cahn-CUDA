include(GoogleTest)

# ── Unit tests ──────────────────────────────────────────────────────────────
add_executable(unit_tests
    # Core (CPU-only)
    unit/test_Grid.cpp
    unit/test_SimulationConfig.cpp
    unit/test_FieldData.cpp
    unit/test_CheckpointIO.cpp
    unit/test_LRUCache.cpp
    unit/test_SpatialHash.cpp
    unit/test_ConcurrentMap.cpp
    unit/test_VTKWriter.cpp
    # CUDA
    unit/test_DeviceField.cu
    unit/test_Laplacian.cu
    unit/test_Anisotropy.cu
    unit/test_BoundaryConditions.cu
    unit/test_Reduction.cu
    unit/test_CudaSolver.cu
    unit/test_MultiGPUSolver.cu
    unit/test_ThermalKernels.cu
    unit/test_Gradient.cu
    unit/test_SimulationEngine.cu
    unit/test_InitialCondition.cu
    unit/test_GridValidation.cu
    unit/test_CheckpointManager.cpp
    unit/test_Logger.cpp
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
    ac_engine
    ac_core
    ac_cuda
    ac_io
    ac_logging
)

target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(unit_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# TEST_PREFIX makes the suite selectable with `ctest -R "^unit[.]"`; without it
# ctest names are bare `Suite.Case` and a `-R unit_tests` filter selects nothing.
gtest_discover_tests(unit_tests
    TEST_PREFIX "unit."
    PROPERTIES TIMEOUT 120
)

# ── Integration tests ───────────────────────────────────────────────────────
add_executable(integration_tests
    integration/test_SphereRegression.cu
    integration/test_EulerConvergence.cu
    integration/test_EnergyConservation.cu
    integration/test_SchemeComparison.cu
    integration/test_CheckpointRestart.cu
    integration/test_StencilComparison.cu
    integration/test_AnisotropyForce.cu
)

target_link_libraries(integration_tests PRIVATE
    GTest::gtest_main
    ac_engine
    ac_core
    ac_cuda
    ac_io
    ac_logging
)

target_include_directories(integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(integration_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

gtest_discover_tests(integration_tests
    TEST_PREFIX "integration."
    PROPERTIES TIMEOUT 300
)

# ── End-to-end tests (the real binary) ──────────────────────────────────────
# Exit code 77 = no GPU (reported as Skipped; signal_exit.py fails instead
# when AC_REQUIRE_GPU=1).
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME e2e.SignalExitStatus
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/signal_exit.py
            $<TARGET_FILE:allen-cahn-cuda> ${CMAKE_CURRENT_BINARY_DIR}/e2e_signal_exit)
set_tests_properties(e2e.SignalExitStatus PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 600
)
