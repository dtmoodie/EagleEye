find_package(OpenCV QUIET COMPONENTS tracking)

if(OpenCV_FOUND)
file(GLOB_RECURSE node_src "src/*.cpp" "src/*.hpp")
set(GPU_TRACKING_SRC
    "dependencies/GPU-tracking/src/dft.cu"
    "dependencies/GPU-tracking/src/trackerKCFparallel.cu"
    "dependencies/GPU-tracking/src/trackerKCFparallel.hpp"
    "dependencies/GPU-tracking/src/trackerKCF.cpp"
    "dependencies/GPU-tracking/src/mulspectrums.cu"
    "dependencies/GPU-tracking/src/featureColorName.cpp"
    "dependencies/GPU-tracking/src/cycleTimer.h"
)

cuda_add_library(aqgputracking SHARED ${node_src} ${GPU_TRACKING_SRC})

target_link_libraries(aqgputracking
    aqcore
)

set_plugin_available(aqgputracking)
set_plugin_status(aqgputracking "Found opencv tracking module")

else()
    set_plugin_unavailable(aqgputracking)
    set_plugin_status(aqgputracking "Unable to find opencv tracking module")
endif()
