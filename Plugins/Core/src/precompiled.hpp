#pragma once

#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include "Aquila/core/IGraph.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/utilities/cuda/CudaUtils.hpp>
#include "Aquila/utilities/cuda/CudaCallbacks.hpp"
#include "Aquila/rcc/SystemTable.hpp"
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/Types.hpp>
#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <Aquila/rcc/external_includes/cv_core.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <Aquila/rcc/external_includes/cv_highgui.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <Aquila/rcc/external_includes/cv_cudabgsegm.hpp>
#include <Aquila/rcc/external_includes/cv_cudafeatures2d.hpp>
#include <Aquila/rcc/external_includes/cv_cudafilters.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudalegacy.hpp>
#include <Aquila/rcc/external_includes/cv_cudaobjdetect.hpp>
#include <Aquila/rcc/external_includes/cv_cudaoptflow.hpp>
#include <opencv2/core/opengl.hpp>

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <utility>
