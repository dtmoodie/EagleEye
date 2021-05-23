#include <gtest/gtest.h>

#include <ct/types/opencv.hpp>

#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/TPublisher.hpp>

#include <aqframegrabbers_export.hpp>

#include <boost/thread.hpp>


TEST(gstreamer, construct_dynamic)
{
    rcc::shared_ptr<aq::nodes::IGrabber> obj =
        mo::MetaObjectFactory::instance()->create("GrabberGstreamer");
    ASSERT_TRUE(obj);
}

TEST(gstreamer, videotestsrc)
{
    auto stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    auto fg = aq::nodes::IFrameGrabber::create("videotestsrc ! appsink", "frame_grabber_gstreamer");
    ASSERT_TRUE(fg);
    mo::IPublisher* output = fg->getOutput("output");
    ASSERT_TRUE(output);
    mo::TPublisher<aq::SyncedImage>* typed = dynamic_cast<mo::TPublisher<aq::SyncedImage>*>(output);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    fg->process(*stream);
    ASSERT_TRUE(typed);
    auto data = typed->getData();
    ASSERT_TRUE(data != nullptr);
    auto tdata = std::dynamic_pointer_cast<const mo::TDataContainer<aq::SyncedImage>>(data);
    ASSERT_TRUE(tdata != nullptr);
    ASSERT_TRUE(!tdata->data.empty());
}

int main(int argc, char** argv)
{
    auto table = SystemTable::instance();
    auto factory = mo::MetaObjectFactory::instance(table.get());
    aqframegrabbers::initPlugin(0, factory.get());
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
