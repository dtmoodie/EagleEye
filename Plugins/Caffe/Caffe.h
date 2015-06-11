#include "caffe/caffe.hpp"
//#include "EagleLib/include/nodes/Node.h"

#include "nodes/Node.h"

#ifdef __cplusplus
extern "C"{
#endif
    CV_EXPORTS IPerModuleInterface* CALL GetModule();
    CV_EXPORTS void CALL SetupIncludes();

#ifdef __cplusplus
}
#endif
