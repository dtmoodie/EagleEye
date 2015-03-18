#pragma once

#include "nodes/Node.h"
//#include <opencv2/highgui.hpp>
//RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_highgui")

namespace EagleLib
{
    class TestNode: public Node
    {
    public:
        TestNode():Node()
        {
            nodeName = "TestNode";
            treeName = nodeName;
            //parent = nullptr;
        }

        ~TestNode(){}

        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
            std::cout << "test" << std::endl;
            //cv::imshow("Test", cv::Mat(img));
            return img;
        }

    };


}
using namespace EagleLib;
REGISTERCLASS(TestNode)


