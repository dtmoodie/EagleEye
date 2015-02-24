#include <nodes/Node.h>
#include <../RuntimeObjectSystem/ISimpleSerializer.h>
#include <opencv2/highgui.hpp>
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
namespace EagleLib
{
    class TestNode: public TInterface<IID_NodeObject, Node>
    {
    public:
        TestNode()
        {
            nodeName = "TestNode";
            treeName = nodeName;
            //Init(true);
        }
        virtual ~TestNode()
        {
            std::cout << "Deleting  testNode" << std::endl;
        }
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
            cv::Mat h_img = cv::imread("/home/dan/Dropbox/Photos/x0ml8.png");
            std::cout << h_img.size() << std::endl;
            return cv::cuda::GpuMat(h_img);
            std::cout << getParameter<std::string>("Output")->data << std::endl;
            std::cout <<"Beeeyah!" << std::endl;
           // std::cout << "Test!" << std::endl;
            return img;
        }
        virtual void Init(bool firstInit)
        {
            std::cout << "Initializing TestNode with firstInit: " << firstInit << std::endl;
            if(firstInit)
                addParameter("Output", std::string("Defaultasdfaasdfsdf!asdf!!!!!!!"));
        }
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            std::cout << "Running TestNode Serializer" << std::endl;
            Node::Serialize(pSerializer);
        }

    private:

    };

    class TestChildNode: public TInterface<IID_NodeObject, Node>
    {
    public:
        TestChildNode()
        {
            nodeName = "TestChildNode";
            treeName = "TestChildNode";
            //addParameter("Output", std::string("DefaultX"));
        }
        virtual ~TestChildNode() {}
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
           // std::cout << getParameter<std::string>("Output")->data << std::endl;
            cv::imshow("test", cv::Mat(img));
            std::cout << "ChildNode!" << std::endl;
            return img;
        }

    private:

    };

}
using namespace EagleLib;
REGISTERCLASS(TestNode)
REGISTERCLASS(TestChildNode)
