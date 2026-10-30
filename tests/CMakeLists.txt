include(GoogleTest)

# ── Unit tests ──────────────────────────────────────────────────────────────
add_executable(unit_tests
    # Core (CPU-only)
    unit/test_Grid.cpp
    unit/test_SimulationConfig.cpp
    unit/test_StabilityLimits.cpp
    unit/test_LiquidModes.cpp
    unit/test_KernelRanges.cpp
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
    unit/test_ImexJacobi.cu
    unit/test_StabilityLimitsGpu.cu
    unit/test_MultiGPUSolver.cu
    unit/test_Willmore.cu
    unit/test_WillmorePreconditioner.cu
    unit/test_ThermalKernels.cu
    unit/test_Gradient.cu
    unit/test_SimulationEngine.cu
    unit/test_InitialCondition.cu
    unit/test_GridValidation.cu
    unit/test_CheckpointManager.cpp
    unit/test_Logger.cpp
)

# quadmath: binary128 sin, cos and sqrt for the reference values of
# common/LiquidModeReference.hpp (part of GCC's runtime).
target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
    ac_engine
    ac_core
    ac_cuda
    ac_io
    ac_logging
    quadmath
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
    integration/test_WillmoreRefinement.cu
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

# ── End-to-end tests (the real binary, the Makefile) ────────────────────────
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

# A retry of the shipped Kubernetes Job (its container arguments, k8s/base/job.yaml)
# continues the interrupted run from its checkpoint instead of starting over.
add_test(NAME e2e.KubernetesJobResumes
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/k8s_job_resume.py
            $<TARGET_FILE:allen-cahn-cuda> ${PROJECT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/e2e_k8s_job_resume)
set_tests_properties(e2e.KubernetesJobResumes PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 600
)

# Only the documented command lines are accepted; any other argument is an error,
# not ignored (no GPU needed: every case ends before a run).
add_test(NAME e2e.CommandLineArguments
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/cli_arguments.py
            $<TARGET_FILE:allen-cahn-cuda> ${CMAKE_CURRENT_BINARY_DIR}/e2e_cli_arguments)
set_tests_properties(e2e.CommandLineArguments PROPERTIES TIMEOUT 120)

# Every configuration the repository ships (config/*.json, k8s payloads, the
# Makefile's run_vtk.json, README/docs examples) must pass --validate-config.
add_test(NAME e2e.ShippedConfigsValidate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/validate_shipped_configs.py
            $<TARGET_FILE:allen-cahn-cuda> ${PROJECT_SOURCE_DIR}
)
set_tests_properties(e2e.ShippedConfigsValidate PROPERTIES TIMEOUT 300)

# The Makefile's resume targets must pass the resume helper only the config, so
# that it searches the config's own checkpoint_dir. Runs `make -n` with a stub
# helper (GNU make on PATH; no binary, docker or GPU).
add_test(NAME e2e.MakefileResumeCall
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/makefile_resume_call.py
            ${PROJECT_SOURCE_DIR}
)
set_tests_properties(e2e.MakefileResumeCall PROPERTIES TIMEOUT 60)

# `make cuda-visualize` renders a run into viz/<name of the run> however VIZ_IN
# is spelled (trailing or repeated slashes, '.'). Runs `make -n` (GNU make on
# PATH; no binary, docker or GPU).
add_test(NAME e2e.MakefileVizOut
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e/makefile_viz_out.py
            ${PROJECT_SOURCE_DIR}
)
set_tests_properties(e2e.MakefileVizOut PROPERTIES TIMEOUT 60)
