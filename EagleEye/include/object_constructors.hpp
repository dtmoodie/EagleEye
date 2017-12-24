#pragma once
#include "NodeData.hpp"
#include "NodeDataModel.hpp"
struct IObjectConstructor;

namespace aq{
    namespace nodes{
        class Node;
    }
class NodeConstructor : virtual public QtNodes::NodeDataModel {
public:
    NodeConstructor(IObjectConstructor* constructor);
    virtual QString caption() const;
    virtual QString name() const;
    virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const;

    virtual unsigned int nPorts(QtNodes::PortType portType) const;

    virtual QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const;

    virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port);

    virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port);
    virtual QWidget * embeddedWidget();

    IObjectConstructor* m_constructor;
};

class GraphConstructor : virtual public QtNodes::NodeDataModel {
public:
    GraphConstructor();
    virtual QString caption() const;
    virtual QString name() const;
    virtual unsigned int nPorts(QtNodes::PortType portType) const;
    virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const;
    virtual QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const;
    virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port);
    virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port);
    virtual QWidget * embeddedWidget();
};

class FrameGrabberConstructor : virtual public NodeConstructor {
public:
    FrameGrabberConstructor(IObjectConstructor* constructor);
    virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const;
};
} // namespace aq