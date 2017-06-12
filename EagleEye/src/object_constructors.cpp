#include "object_constructors.hpp"
#include "object_proxies.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/core/IDataStream.hpp"
#include "RuntimeObjectSystem/ObjectInterface.h"
using namespace aq;

NodeConstructor::NodeConstructor(IObjectConstructor* constructor) : m_constructor(constructor) {}
QString NodeConstructor::caption() const { return m_constructor->GetName(); }
QString NodeConstructor::name() const { return caption(); }
std::unique_ptr<QtNodes::NodeDataModel> NodeConstructor::clone() const {
    rcc::shared_ptr<aq::Nodes::Node> node(m_constructor->Construct());
    node->Init(true);
    return std::unique_ptr<QtNodes::NodeDataModel>(new NodeProxy(std::move(node)));
}


unsigned int NodeConstructor::nPorts(QtNodes::PortType portType) const {
    return 0;
}


QtNodes::NodeDataType NodeConstructor::dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const {
    return QtNodes::NodeDataType();
}


void NodeConstructor::setInData(std::shared_ptr<QtNodes::NodeData> nodeData,
    QtNodes::PortIndex port) {
    // TODO
}


std::shared_ptr<QtNodes::NodeData> NodeConstructor::outData(QtNodes::PortIndex port) {
    return std::shared_ptr<QtNodes::NodeData>();
}

QWidget * NodeConstructor::embeddedWidget() {
    // TODO
    return nullptr;
}

DataStreamConstructor::DataStreamConstructor() {}
QString DataStreamConstructor::caption() const { return "DataStream"; }
QString DataStreamConstructor::name() const { return "DataStream"; }
unsigned int DataStreamConstructor::nPorts(QtNodes::PortType portType) const {
    if (portType == QtNodes::PortType::Out) return 1;
    return 0;
}
std::unique_ptr<QtNodes::NodeDataModel> DataStreamConstructor::clone() const {
    return std::unique_ptr<DataStreamProxy>(new DataStreamProxy(std::move(aq::IDataStream::create("", ""))));
}

QtNodes::NodeDataType DataStreamConstructor::dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const {
    QtNodes::NodeDataType output;
    if (port_type == QtNodes::PortType::Out) {
        output.id = "aq::Nodes::Node";
        output.name = "children";
    }
    return output;
}

void DataStreamConstructor::setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port) {

}

std::shared_ptr<QtNodes::NodeData> DataStreamConstructor::outData(QtNodes::PortIndex port) {
    return std::shared_ptr<QtNodes::NodeData>();
}

QWidget * DataStreamConstructor::embeddedWidget() {
    // TODO
    return nullptr;
}

FrameGrabberConstructor::FrameGrabberConstructor(IObjectConstructor* constructor) : NodeConstructor(constructor) {}
std::unique_ptr<QtNodes::NodeDataModel> FrameGrabberConstructor::clone() const {
    rcc::shared_ptr<aq::Nodes::Node> node(m_constructor->Construct());
    node->Init(true);
    return std::unique_ptr<QtNodes::NodeDataModel>(new FrameGrabberProxy(std::move(node)));
}