cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(VERSION 2.8) 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME geomRenderer)

project(${NAME})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


find_package(VTK
  COMPONENTS
    FiltersGeometry
    IOExport
    IOGeometry
    IOImage
    IOLegacy
    CommonCore
    CommonDataModel
    CommonTransforms
    CommonExecutionModel
    CommonSystem
    CommonMisc
    FiltersCore
    FiltersGeneral
    FiltersModeling
    IOXML
    ImagingCore
    ImagingStatistics
    InteractionStyle
    RenderingCore
    RenderingVolume
    RenderingOpenGL2
    RenderingVolumeOpenGL2
  OPTIONAL_COMPONENTS
    TestingCore
    TestingRendering)
if (NOT VTK_FOUND)
  message("Skipping example: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif ()

set(include_dirs
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )
include_directories(${include_dirs})

IF(WIN32)

find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs)
IF(Vulkan_LIBRARY)
set(VULKAN_FOUND ON)
MESSAGE("Using bundled Vulkan library version")
ENDIF()

ENDIF(WIN32)

find_library(glfw_LIBRARY NAMES glfw3.lib PATHS "C:/Users/lenovo/Documents/Visual-Studio-2019/Libraries/glfw-3.3.2.bin.WIN64/lib-vc2015")
IF(glfw_LIBRARY)
MESSAGE("Using glfw library version 2015")
ENDIF()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_CXX_STANDARD 17)	
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(KTX_DIR "C:/Users/lenovo/Documents/Visual-Studio-2019/Axillary-Code/gltfModel/gltfModel-Master/external/ktx")
set(KTX_SOURCES 
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c)

include_directories(Includes)
set(vulkan_directory "C:/VulkanSDK/1.2.135.0/Include")
set(glm_directory "${CMAKE_SOURCE_DIR}/external/glm")
set(glfw_directory "C:/Users/lenovo/Documents/Visual-Studio-2019/Libraries/glfw-3.3.2.bin.WIN64/include")
include_directories(${glm_directory})
include_directories(${vulkan_directory})
include_directories("${CMAKE_SOURCE_DIR}/external/tinygltf")
include_directories("${CMAKE_SOURCE_DIR}/external/imgui")
include_directories(${glfw_directory})


file(GLOB IMG_SOURCES "${CMAKE_SOURCE_DIR}/external/imgui/*.cpp")

add_library(gltfModel gltfModel.cpp ${KTX_SOURCES} ${IMG_SOURCES})


target_link_libraries(gltfModel ${Vulkan_LIBRARY} ${glfw_LIBRARY}) # removed winlibs


target_include_directories(gltfModel PUBLIC "${CMAKE_SOURCE_DIR}/external/ktx/other_include")

set(EXAMPLE_NAME geomRenderer)

if(WIN32)
add_executable(${EXAMPLE_NAME} WIN32 main.cpp) 
# removed winlibs
target_link_libraries(${EXAMPLE_NAME} ${Vulkan_LIBRARY} gltfModel ${ITK_LIBRARIES} ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS ${EXAMPLE_NAME}
  MODULES ${VTK_LIBRARIES})
endif(WIN32)



 # Compiler specific stuff
IF(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF(MSVC)










