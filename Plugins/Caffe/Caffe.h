#include "caffe/caffe.hpp"

#include "nodes/Node.h"

#ifdef __cplusplus
extern "C"{
#endif
    CV_EXPORTS IPerModuleInterface* GetModule();
    CV_EXPORTS void SetupIncludes();

#ifdef __cplusplus
}
#endif
