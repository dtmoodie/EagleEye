#pragma once
#include "NodeData.hpp"
#include "NodeDataModel.hpp"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include "MetaObject/signals/TSlot.hpp"
class QComboBox;
namespace mo{
    class IParam;
    namespace UI{
        namespace qt{ class IParamProxy; }
    }
}

namespace aq{

class IGraph;
class Algorithm;
namespace nodes{
    class Node;
    class IFrameGrabber;
}

class NodeOutProxy : virtual public QtNodes::NodeData {
public:
    NodeOutProxy(const rcc::shared_ptr<nodes::Node>& node_);
    virtual QtNodes::NodeDataType type() const;
    rcc::shared_ptr<nodes::Node> node;
};

class DSOutProxy : virtual public QtNodes::NodeData {
public:
    DSOutProxy(const rcc::shared_ptr<IGraph>& ds);
    virtual QtNodes::NodeDataType type() const;
    rcc::shared_ptr<IGraph> m_ds;
};


class ParamOutProxy : virtual public NodeOutProxy {
public:
    ParamOutProxy(mo::IParam* param_, const rcc::shared_ptr<nodes::Node>& node_);
    virtual QtNodes::NodeDataType type() const;
    mo::IParam* param;
};

class NodeProxy : public QtNodes::NodeDataModel {
public:
    NodeProxy(rcc::shared_ptr<aq::nodes::Node>&& obj);
    NodeProxy(const rcc::shared_ptr<aq::nodes::Node>& obj);
    virtual QString caption() const;
    virtual QString name() const;
    virtual std::unique_ptr<QtNodes::NodeDataModel> clone()const;

    virtual unsigned int nPorts(QtNodes::PortType portType) const;

    virtual QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const;

    virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port);

    virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port);
    virtual QWidget * embeddedWidget();

    rcc::shared_ptr<aq::nodes::Node> m_obj;
    std::vector<std::shared_ptr<mo::UI::qt::IParamProxy>> m_param_proxies;
    QWidget* widget = nullptr;
protected:
    virtual void onComponentAdded(aq::Algorithm* cpt);
private:
    mo::TSlot<void(aq::Algorithm*)> m_slot_component_added;
};


class GraphProxy : virtual public QtNodes::NodeDataModel {
public:
    GraphProxy(rcc::shared_ptr<aq::IGraph>&& ds = rcc::shared_ptr<aq::IGraph>());
    GraphProxy(const rcc::shared_ptr<aq::IGraph>& ds);

    virtual QString caption() const;
    virtual QString name() const;
    virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const;
    virtual unsigned int nPorts(QtNodes::PortType portType) const;
    virtual QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const;
    virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port);
    virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port);
    virtual QWidget * embeddedWidget();

    rcc::shared_ptr<aq::IGraph> m_obj;
};


class FrameGrabberProxy : public NodeProxy {
    Q_OBJECT
public:
    FrameGrabberProxy(rcc::shared_ptr<aq::nodes::Node>&& obj);
    FrameGrabberProxy(const rcc::shared_ptr<aq::nodes::Node>& obj);
    virtual QWidget * embeddedWidget();
public slots:
    void onSelectionChanged();
    void onSelectionChanged(int idx);
protected:
    rcc::shared_ptr<aq::nodes::IFrameGrabber> m_fg;
    QComboBox* selector = nullptr;
};
} // namespace aq