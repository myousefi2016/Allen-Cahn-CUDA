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
    unit/test_ThermalKernels.cu
    unit/test_Gradient.cu
    unit/test_SimulationEngine.cu
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

set_target_properties(unit_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

gtest_discover_tests(unit_tests)

# ── Integration tests ───────────────────────────────────────────────────────
add_executable(integration_tests
    integration/test_SphereRegression.cu
    integration/test_EulerConvergence.cu
    integration/test_EnergyConservation.cu
    integration/test_SchemeComparison.cu
    integration/test_CheckpointRestart.cu
    integration/test_StencilComparison.cu
)

target_link_libraries(integration_tests PRIVATE
    GTest::gtest_main
    ac_engine
    ac_core
    ac_cuda
    ac_io
    ac_logging
)

set_target_properties(integration_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

gtest_discover_tests(integration_tests)
