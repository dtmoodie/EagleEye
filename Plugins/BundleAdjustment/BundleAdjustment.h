#pragma once

#include "nodes/Node.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C"{
#endif

	CV_EXPORTS IPerModuleInterface* GetModule();

#ifdef __cplusplus
}
#endif