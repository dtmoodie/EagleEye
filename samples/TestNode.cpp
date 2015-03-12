#include <nodes/Node.h>
#include <../RuntimeObjectSystem/ISimpleSerializer.h>
#include <opencv2/highgui.hpp>
#include <opencv2/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>



RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
namespace EagleLib
{
    class TestNode: public Node
    {
    public:
		TestNode();
        virtual ~TestNode()
        {
            std::cout << "Deleting  testNode " << std::endl;
        }
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
            
            cv::Mat h_img = cv::imread("/home/dan/Dropbox/Photos/x0ml8.png");
            std::cout << h_img.size() << std::endl;
            return cv::cuda::GpuMat(h_img);
           // std::cout << "Test!" << std::endl;
            return img;
        }
        virtual void Init(bool firstInit)
        {
            Node::Init(firstInit);
            std::cout << "Initializing TestNode with firstInit: " << firstInit << std::endl;
			if (firstInit)
			{
				addParameter("Output", std::string("Defautasdfadf!asdf!!!!!!!"));
				addParameter("Output", &testVector, Parameter::Output, "Test output vector", false);
			}
                
        }
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            std::cout << "Running TestNode Serializer" << std::endl;
            Node::Serialize(pSerializer);
        }

    private:
		std::vector<int> testVector;
    };

    class TestChildNode: public Node
    {
    public:
		TestChildNode();
        virtual ~TestChildNode()
        {
            std::cout << "Deleting TestChildNode" << std::endl;
        }
		virtual void Init(bool firstInit)
		{
			Node::Init(firstInit);
			if (firstInit)
			{
				addInputParameter<std::vector<int>>("Test input Vector", "Test input Vector");
			}
		}
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img)
        {
			cv::cuda::resize(img, img, cv::Size(1000, 1000));
           // std::cout << getParameter<std::string>("Output")->data << std::endl;
			std::cout << img.size() << std::endl;
            std::stringstream ss;
            ss << "TestChildNodeDisplay: " << img.size();
            cv::imshow(ss.str(), cv::Mat(img));
			cv::waitKey(30);
            std::cout << "ChildNode!" << std::endl;
            return img;
        }

    private:

    };

}
using namespace EagleLib;
/*
REGISTERCLASS(TestNode)
REGISTERCLASS(TestChildNode)
*/

NODE_DEFAULT_CONSTRUCTOR_IMPL(TestNode);
NODE_DEFAULT_CONSTRUCTOR_IMPL(TestChildNode);