#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/ThreadedNode.h"
#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "EagleLib/Logging.h"
#include "EagleLib/Nodes/FrameGrabberInfo.hpp"
#include "EagleLib/ICoordinateManager.h"


#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/MetaObjectFactory.hpp"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibFrameGrabbers"
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>


using namespace EagleLib;
using namespace EagleLib::Nodes;

struct test_framegrabber: public IFrameGrabber
{
    bool ProcessImpl()
    {
        current.create(128,128, CV_8U);
        ++ts;
        current.setTo(ts);
        current_frame_param.UpdateData(current.clone(), ts, _ctx);
        return true;
    }
    bool LoadFile(const std::string&)
    {
        return true;
    }
    long long GetFrameNumber()
    {
        return ts;
    }
    long long GetNumFrames()
    {
        return 255;
    }
    TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream)
    {
        return TS<SyncedMemory>(current);
    }
    TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream)
    {
        ++ts;
        Process();
        return GetCurrentFrame(stream);
    }
    TS<SyncedMemory> GetFrame(int ts, cv::cuda::Stream& stream)
    {
        cv::Mat output;
        output.create(128,128, CV_8U);
        output.setTo(ts);
        return output;
    }
    TS<SyncedMemory> GetFrameRelative(int offset, cv::cuda::Stream& stream)
    {
        cv::Mat output;
        output.create(128,128, CV_8U);
        output.setTo(ts + offset);
        return output;
    }
    rcc::shared_ptr<EagleLib::ICoordinateManager> GetCoordinateManager()
    {
        return rcc::shared_ptr<EagleLib::ICoordinateManager>();
    }

    MO_DERIVE(test_framegrabber, IFrameGrabber);
    MO_END;
    int ts = 0;
    cv::Mat current;
    
    static int CanLoadDocument(const std::string& doc)
    {
        return 1;
    }
    static int LoadTimeout()
    {
        return 1;
    }
};

struct img_node: public Node
{
    MO_DERIVE(img_node, Node);
        INPUT(SyncedMemory, input, nullptr);
    MO_END;

    bool ProcessImpl()
    {
        BOOST_REQUIRE(input);
        auto mat = input->GetMat(Stream());
        BOOST_REQUIRE_EQUAL(mat.at<uchar>(0), input_param.GetTimestamp());
        return true;
    }
};

MO_REGISTER_CLASS(test_framegrabber);
MO_REGISTER_CLASS(img_node);

BOOST_AUTO_TEST_CASE(test_dummy_output)
{
    EagleLib::SetupLogging();
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
    mo::MetaObjectFactory::Instance()->LoadPlugins("");
    auto info = mo::MetaObjectFactory::Instance()->GetObjectInfo("test_framegrabber");
    BOOST_REQUIRE(info);
    auto fg_info = dynamic_cast<EagleLib::Nodes::FrameGrabberInfo*>(info);
    BOOST_REQUIRE(fg_info);
    std::cout << fg_info->Print();
    
    auto fg = rcc::shared_ptr<test_framegrabber>::Create();
    auto node = rcc::shared_ptr<img_node>::Create();
    BOOST_REQUIRE(node->ConnectInput(fg, "input", "current_frame"));
    for(int i = 0; i < 100; ++i)
    {
        fg->Process();
    }
}
BOOST_AUTO_TEST_CASE(test_enumeration)
{
    //auto all_docs = EagleLib::Nodes::IFrameGrabber::ListAllLoadableDocuments();
    std::cout << mo::MetaObjectFactory::Instance()->PrintAllObjectInfo();
}
