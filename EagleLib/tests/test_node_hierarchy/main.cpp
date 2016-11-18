#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "EagleLib/IDataStream.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/ThreadedNode.h"
#include "EagleLib/Nodes/NodeInfo.hpp"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/MetaObjectFactory.hpp"

#include "EagleLib/Detail/AlgorithmImpl.hpp"
#include "EagleLib/Logging.h"
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibNodes"

#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace EagleLib;
using namespace EagleLib::Nodes;

struct node_a: public Nodes::Node
{
    MO_BEGIN(node_a)
        OUTPUT(int, out_a, 0);
    MO_END;

    bool ProcessImpl()
    {
        ++ts;
        out_a_param.UpdateData(ts, ts);
        _modified = true;
        return true;
    }
    int ts = 0;
};

struct node_b: public Nodes::Node
{
    MO_BEGIN(node_b)
        OUTPUT(int, out_b, 0);
    MO_END;

    bool ProcessImpl()
    {
        ++ts;
        out_b_param.UpdateData(ts, ts);
        _modified = true;
        return true;
    }
    int ts = 0;
};

struct node_c: public Nodes::Node
{
    MO_BEGIN(node_c)
        INPUT(int, in_a, nullptr);
        INPUT(int, in_b, nullptr);
    MO_END;

    bool ProcessImpl()
    {
        BOOST_REQUIRE_EQUAL(*in_a, *in_b);
        count += *in_a + *in_b;
        return true;
    }
    void check_timestamps()
    {
        Algorithm::impl* impl = _pimpl;
        LOG(debug) << impl->_ts_processing_queue.size() << " frames left to process";
    }
    int count = 0;
};

MO_REGISTER_CLASS(node_a);
MO_REGISTER_CLASS(node_b);
MO_REGISTER_CLASS(node_c);

BOOST_AUTO_TEST_CASE(branching)
{
    EagleLib::SetupLogging();
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
    
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    a->AddChild(b);
    a->Process();
    BOOST_REQUIRE_EQUAL(c->count, 2);
}


BOOST_AUTO_TEST_CASE(merging)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    a->Process();
    BOOST_REQUIRE_EQUAL(c->count, 0);
    b->Process();
    BOOST_REQUIRE_EQUAL(c->count, 2);
}


BOOST_AUTO_TEST_CASE(diamond)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b1 = rcc::shared_ptr<node_b>::Create();
    auto b2 = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    // B1 and B2 don't have inputs, thus for them to be a child of A, we need to manually add them
    a->AddChild(b1);
    a->AddChild(b2);
    // C is added as a child of B1 and B2 here
    BOOST_REQUIRE(c->ConnectInput(b1, "in_a", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(b2, "in_b", "out_b"));
    // Calling A, calls B1, then B2.  When B1 Process() is called, it tries to call C->Process()
    // Since C's inputs aren't ready yet, it is not called, but when B2 Process is called, C->Process() does get
    // called since B2 finished prepping the inputs for C.

    a->Process();
    BOOST_REQUIRE_EQUAL(c->count, 2);
}

BOOST_AUTO_TEST_CASE(threaded_child_sync_every)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    auto thread = rcc::shared_ptr<EagleLib::Nodes::ThreadedNode>::Create();
    mo::Context ctx;
    a->SetContext(&ctx);
    b->SetContext(&ctx);
    thread->AddChild(c);
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));

    c->SetSyncInput("in_b");
    int sum = 0;

    a->Process();
    b->Process();
    sum += a->out_a;
    sum += b->out_b;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    BOOST_REQUIRE_EQUAL(c->count, 2);
    
    for(int i = 0; i < 100; ++i)
    {
        a->Process();
        b->Process();
        sum += a->out_a;
        sum += b->out_b;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    thread->StopThread();
    if(c->count != sum)
    {
        c->check_timestamps();
    }
    std::cout << "Dropped " << 1.0 -  double(c->count) / (double)sum << " % of data\n";
    BOOST_REQUIRE_EQUAL(c->count, sum);
}

BOOST_AUTO_TEST_CASE(threaded_child_sync_newest)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    auto thread = rcc::shared_ptr<EagleLib::Nodes::ThreadedNode>::Create();
    mo::Context ctx;
    a->SetContext(&ctx);
    b->SetContext(&ctx);
    thread->AddChild(c);
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));

    c->SetSyncInput("in_b");
    c->SetSyncMethod(Algorithm::SyncNewest);
    int sum = 0;

    a->Process();
    b->Process();
    sum += a->out_a;
    sum += b->out_b;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    BOOST_REQUIRE_EQUAL(c->count, 2);

    for(int i = 0; i < 100; ++i)
    {
        a->Process();
        b->Process();
        sum += a->out_a;
        sum += b->out_b;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    thread->StopThread();
    BOOST_REQUIRE_LE(c->count, sum);
    std::cout << "Dropped " << 1.0 -  double(c->count) / (double)sum << " % of data\n";
}


BOOST_AUTO_TEST_CASE(threaded_parent)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
}

