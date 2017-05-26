#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Aquila/frame_grabbers/test_gstreamer"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/ITParam.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
BOOST_AUTO_TEST_CASE(gstreamer_load)
{
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    mo::MetaObjectFactory::instance()->loadPlugins("");
    auto plugins = mo::MetaObjectFactory::instance()->listLoadedPlugins();
    bool found = false;
    for(const auto& plugin : plugins)
    {
        if(plugin.find_first_of("frame_grabber") != std::string::npos)
        {
            if(plugin.find_first_of("success") != std::string::npos)
                found = true;
        }
    }
    BOOST_REQUIRE(found);
}

BOOST_AUTO_TEST_CASE(gstreamer_construct_static)
{
    
}

BOOST_AUTO_TEST_CASE(gstreamer_construct_dynamic)
{
    rcc::shared_ptr<aq::Nodes::IFrameGrabber> obj = mo::MetaObjectFactory::instance()->create("frame_grabber_gstreamer");
    BOOST_REQUIRE(obj);
}

BOOST_AUTO_TEST_CASE(gstreamer_videotestsrc)
{
    auto fg = aq::Nodes::IFrameGrabber::create("videotestsrc ! appsink", "frame_grabber_gstreamer");
    BOOST_REQUIRE(fg);
    auto output = fg->getOutput("current_frame");
    mo::ITParam<aq::SyncedMemory>* typed = dynamic_cast<mo::ITParam<aq::SyncedMemory>*>(output);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    fg->process();
    BOOST_REQUIRE(!typed->GetDataPtr()->empty());
}
