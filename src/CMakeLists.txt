# ── Logging library ─────────────────────────────────────────────────────────
add_library(ac_logging STATIC
    logging/Logger.cpp
)
target_include_directories(ac_logging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ac_logging PUBLIC spdlog::spdlog)

# ── Core library (no CUDA dependency) ──────────────────────────────────────
add_library(ac_core STATIC
    core/SimulationConfig.cpp
    core/Grid.cpp
    core/FieldData.cpp
)
target_include_directories(ac_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ac_core
    PUBLIC
        nlohmann_json::nlohmann_json
        ac_logging
)

# ── CUDA library ────────────────────────────────────────────────────────────
add_library(ac_cuda STATIC
    cuda/CudaUtils.cu
    cuda/DeviceField.cu
    cuda/AllenCahnKernels.cu
    cuda/ThermalKernels.cu
    cuda/BoundaryKernels.cu
    cuda/ReductionKernels.cu
    cuda/CudaSolver.cu
    cuda/MultiGPUSolver.cu
)
target_include_directories(ac_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ac_cuda PUBLIC ac_core)
set_target_properties(ac_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ── I/O library ─────────────────────────────────────────────────────────────
add_library(ac_io STATIC
    io/VTKWriter.cpp
    io/CheckpointIO.cpp
    core/CheckpointManager.cpp
)
target_include_directories(ac_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ac_io PUBLIC ac_core)

if(AC_HAS_VTK)
    target_link_libraries(ac_io PUBLIC VTK::CommonCore VTK::CommonDataModel VTK::IOXML)
    target_compile_definitions(ac_io PUBLIC AC_HAS_VTK=1)
endif()

if(AC_HAS_HDF5)
    target_link_libraries(ac_io PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(ac_io PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(ac_io PUBLIC AC_HAS_HDF5=1)
endif()

# ── Simulation engine library ──────────────────────────────────────────────
add_library(ac_engine STATIC
    core/SimulationEngine.cu
)
set_target_properties(ac_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(ac_engine PUBLIC ac_core ac_cuda ac_io ac_logging)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(allen-cahn-cuda main.cpp)
target_link_libraries(allen-cahn-cuda PRIVATE ac_engine)
