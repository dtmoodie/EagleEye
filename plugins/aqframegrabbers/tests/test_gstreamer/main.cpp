#include <gtest/gtest.h>

#include <ct/types/opencv.hpp>

#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/TPublisher.hpp>

#include <boost/thread.hpp>

TEST(gstreamer, load)
{
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    mo::MetaObjectFactory::instance()->loadPlugins("");
    auto plugins = mo::MetaObjectFactory::instance()->listLoadedPlugins();
    bool found = false;
    for (const auto& plugin : plugins)
    {
        if (plugin.find_first_of("frame_grabber") != std::string::npos)
        {
            if (plugin.find_first_of("success") != std::string::npos)
                found = true;
        }
    }
    ASSERT_TRUE(found);
}

TEST(gstreamer, construct_dynamic)
{
    rcc::shared_ptr<aq::nodes::IFrameGrabber> obj =
        mo::MetaObjectFactory::instance()->create("frame_grabber_gstreamer");
    ASSERT_TRUE(obj);
}

TEST(gstreamer, videotestsrc)
{
    auto stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    auto fg = aq::nodes::IFrameGrabber::create("videotestsrc ! appsink", "frame_grabber_gstreamer");
    ASSERT_TRUE(fg);
    mo::IPublisher* output = fg->getOutput("current_frame");
    mo::TPublisher<aq::SyncedImage>* typed = dynamic_cast<mo::TPublisher<aq::SyncedImage>*>(output);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    fg->process(*stream);
    auto data = typed->getData();
    ASSERT_TRUE(data != nullptr);
    auto tdata = std::dynamic_pointer_cast<const mo::TDataContainer<aq::SyncedImage>>(data);
    ASSERT_TRUE(tdata != nullptr);
    ASSERT_TRUE(!tdata->data.empty());
}
