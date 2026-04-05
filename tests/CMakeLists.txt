include(GoogleTest)

# ── Unit tests ──────────────────────────────────────────────────────────────
add_executable(unit_tests
    unit/test_Grid.cpp
    unit/test_SimulationConfig.cpp
    unit/test_FieldData.cpp
    unit/test_CheckpointIO.cpp
    unit/test_DeviceField.cu
    unit/test_Laplacian.cu
    unit/test_Anisotropy.cu
    unit/test_BoundaryConditions.cu
    unit/test_Reduction.cu
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
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
