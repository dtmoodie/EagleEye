
CMAKE_POLICY(SET CMP0020 NEW)
find_package(CUDA REQUIRED)

find_package(Ceres QUIET)

set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams chrono regex)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

IF(${Ceres_FOUND})
    include_directories(
        ${CERES_INCLUDE_DIRS}
        ${Aquila_INCLUDE_DIRECTORIES}
    )
    if(RCC_VERBOSE_CONFIG)
        message( STATUS
        "========== Ceres ================
          Include: ${CERES_INCLUDE_DIRS}

          Libs: ${CERES_LIBRARIES}

          Lib dir: ${CERES_LIBRARY_DIR}

          ${CUDA_LIBRARIES}

          ${AquilaES}
        ")
    endif(RCC_VERBOSE_CONFIG)

    LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    STRING(REGEX REPLACE ";" "+" CERES_INCLUDE_DIRS_ "${CERES_INCLUDE_DIRS}+" )
    add_definitions(-DCERES_LIB_DIR="${Ceres_DIR}/../lib")
    add_definitions(-DCERES_INC_DIR="${CERES_INCLUDE_DIRS_}")
    add_library(BundleAdjustment SHARED BundleAdjustment.h BundleAdjustment.cpp)
    target_link_libraries(BundleAdjustment ${Boost_LIBRARIES} ${AquilaES} ${CUDA_LIBRARIES})
    set_target_properties(BundleAdjustment PROPERTIES FOLDER "Plugins")
    add_dependencies(BundleAdjustment Aquila{Ceres_FOUND})
message(STATUS "Ceres:\t\tNotFound")
ENDIF(${Ceres_FOUND})
