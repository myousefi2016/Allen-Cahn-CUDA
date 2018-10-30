CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Allen-Cahn-CUDA)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")
CUDA_ADD_EXECUTABLE(Allen-Cahn-CUDA ${SOURCES})
 
if(VTK_LIBRARIES)
  target_link_libraries(Allen-Cahn-CUDA ${VTK_LIBRARIES})
else()
  target_link_libraries(Allen-Cahn-CUDA vtkHybrid vtkWidgets)
endif()
