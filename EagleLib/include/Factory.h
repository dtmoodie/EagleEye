#pragma once
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>
namespace EagleLib
{
#define REGISTER_TYPE(objectClass)                                  \
class objectClass##Factory : public NodeFactory {                   \
public:                                                             \
    objectClass##Factory(){                                         \
        Node::registerType(#objectClass, this);                     \
    }                                                               \
    virtual boost::shared_ptr<Node> create() {                      \
        return boost::shared_ptr<Node>(new objectClass());          \
    }                                                               \
};                                                                  \
static objectClass##Factory global_##objectClass##Factory;  // Static object forces constructor to be called at startup

class Node;
class NodeFactory
{
public:
    virtual boost::shared_ptr<Node> create() = 0;
};



}
