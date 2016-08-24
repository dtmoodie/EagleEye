#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "EagleLib/nodes/Node.h"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/MetaObjectFactory.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibNodes"

#include <boost/test/unit_test.hpp>
#include <iostream>
using namespace EagleLib;
using namespace EagleLib::Nodes;

struct node_a: public Nodes::Node
{
    MO_BEGIN(node_a)
        OUTPUT(int, out_a, 0);
    MO_END;

    void ProcessImpl()
    {
        ++ts;
        out_a_param.UpdateData(ts, ts);
    }
    int ts = 0;
};

struct node_b: public Nodes::Node
{
    MO_BEGIN(node_b)
        OUTPUT(int, out_b, 0);
    MO_END;

    void ProcessImpl()
    {
        ++ts;
        out_b_param.UpdateData(ts, ts);
    }
    int ts = 0;
};

struct node_c: public Nodes::Node
{
    MO_BEGIN(node_c)
        INPUT(int, in_a, nullptr);
        INPUT(int, in_b, nullptr);
    MO_END;

    void ProcessImpl()
    {
        BOOST_REQUIRE_EQUAL(*in_a, *in_b);
    }
};

MO_REGISTER_CLASS(node_a);
MO_REGISTER_CLASS(node_b);
MO_REGISTER_CLASS(node_c);


BOOST_AUTO_TEST_CASE(no_branching)
{
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    c->ConnectInput(a, "in_a", "out_a");
    a->AddChild(b);
    c->ConnectInput(b, "in_b", "out_b");
    a->Process();
}

BOOST_AUTO_TEST_CASE(branching)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    
    a->Process();
}

BOOST_AUTO_TEST_CASE(diamond)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b1 = rcc::shared_ptr<node_b>::Create();
    auto b2 = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    
    
    a->Process();
}

BOOST_AUTO_TEST_CASE(delete_node)
{

}
