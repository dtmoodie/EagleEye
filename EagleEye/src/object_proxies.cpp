#include "object_proxies.hpp"
#include "MetaObject/params/IParam.hpp"
#include <Aquila/nodes/Node.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/ui/Qt/IParamProxy.hpp>
#include <MetaObject/params/ui/WidgetFactory.hpp>

#include <Aquila/core/IGraph.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/Node.hpp>
#include <QComboBox>
#include <QLayout>
#include <QLineEdit>
#include <qgridlayout.h>

using namespace aq;

NodeOutProxy::NodeOutProxy(const rcc::shared_ptr<nodes::Node>& node_) : node(node_)
{
}

QtNodes::NodeDataType NodeOutProxy::type() const
{
    QtNodes::NodeDataType out;
    out.name = QString::fromStdString(node->getTreeName());
    out.id = QString::fromStdString("aq::nodes::Node");
    return out;
}

DSOutProxy::DSOutProxy(const rcc::shared_ptr<IGraph>& ds) : m_ds(ds)
{
}

QtNodes::NodeDataType DSOutProxy::type() const
{
    QtNodes::NodeDataType out;
    out.name = QString::fromStdString("ds");
    out.id = QString::fromStdString("aq::nodes::Node");
    return out;
}

ParamOutProxy::ParamOutProxy(mo::IParam* param_, const rcc::shared_ptr<nodes::Node>& node_)
    : param(param_), NodeOutProxy(node_)
{
}
QtNodes::NodeDataType ParamOutProxy::type() const
{
    QtNodes::NodeDataType out;
    out.name = QString::fromStdString(param->getName());
    out.id = QString::fromStdString(param->getTypeInfo().name());
    return out;
}

NodeProxy::NodeProxy(rcc::shared_ptr<aq::nodes::Node>&& obj)
    : m_obj(std::move(obj)),
      m_slot_component_added(std::bind(&NodeProxy::onComponentAdded, this, std::placeholders::_1))
{
    m_slot_component_added.connect(m_obj->getSignal_componentAdded<void(aq::IAlgorithm*)>());
}

NodeProxy::NodeProxy(const rcc::shared_ptr<aq::nodes::Node>& obj)
    : m_obj(obj), m_slot_component_added(std::bind(&NodeProxy::onComponentAdded, this, std::placeholders::_1))
{
    m_slot_component_added.connect(m_obj->getSignal_componentAdded<void(aq::IAlgorithm*)>());
}

QString NodeProxy::caption() const
{
    return QString::fromStdString(m_obj->getTreeName());
}
QString NodeProxy::name() const
{
    return caption();
}

std::unique_ptr<QtNodes::NodeDataModel> NodeProxy::clone() const
{
    IObjectConstructor* ctr = m_obj->GetConstructor();
    rcc::shared_ptr<aq::nodes::Node> node(ctr->Construct());
    node->Init(true);
    return std::unique_ptr<QtNodes::NodeDataModel>(new NodeProxy(std::move(node)));
}

unsigned int NodeProxy::nPorts(QtNodes::PortType portType) const
{
    switch (portType)
    {
    case QtNodes::PortType::In:
        return m_obj->getInputs().size() + 1;
    case QtNodes::PortType::Out:
        return m_obj->getOutputs().size() + 1;
    case QtNodes::PortType::None:
        return 0;
    }
    return 0;
}

QtNodes::NodeDataType NodeProxy::dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const
{
    QtNodes::NodeDataType output;
    switch (portType)
    {
    case QtNodes::PortType::In:
    {
        if (portIndex == 0)
        {
            output.id = "aq::nodes::Node";
            output.name = "parents";
        }
        else
        {
            auto inputs = m_obj->getInputs();
            if (portIndex <= inputs.size() && portIndex > 0)
            {
                output.id = QString::fromStdString(inputs[portIndex - 1]->getTypeInfo().name());
                output.name = QString::fromStdString(inputs[portIndex - 1]->getName());
            }
        }
        break;
    }
    case QtNodes::PortType::Out:
    {
        if (portIndex == 0)
        {
            output.id = "aq::nodes::Node";
            output.name = "children";
        }
        else
        {
            auto outputs = m_obj->getOutputs();
            if (portIndex > 0 && portIndex <= outputs.size())
            {
                output.id = QString::fromStdString(outputs[portIndex - 1]->getTypeInfo().name());
                output.name = QString::fromStdString(outputs[portIndex - 1]->getName());
            }
        }
    }
    case QtNodes::PortType::None:
        break;
    }
    return output;
}

void NodeProxy::setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port)
{
    if (port == 0)
    {
        std::shared_ptr<NodeOutProxy> typed = std::dynamic_pointer_cast<NodeOutProxy>(nodeData);
        if (typed)
        {
            m_obj->addParent(typed->node.get());
            return;
        }
        std::shared_ptr<DSOutProxy> ds = std::dynamic_pointer_cast<DSOutProxy>(nodeData);
        if (ds)
        {
            ds->m_ds->addNode(m_obj);
        }
    }
    std::shared_ptr<ParamOutProxy> typed = std::dynamic_pointer_cast<ParamOutProxy>(nodeData);
    if (typed)
    {
        auto inputs = m_obj->getInputs();
        if (port > 0 && port <= inputs.size())
        {
            m_obj->connectInput(typed->node, typed->param, inputs[port - 1]);
        }
    }
}

std::shared_ptr<QtNodes::NodeData> NodeProxy::outData(QtNodes::PortIndex port)
{
    if (port == 0)
    {
        return std::make_shared<NodeOutProxy>(m_obj);
    }
    auto outputs = m_obj->getOutputs();
    if (port > 0 && port <= outputs.size())
    {
        return std::make_shared<ParamOutProxy>(outputs[port - 1], m_obj);
    }
    return std::shared_ptr<QtNodes::NodeData>();
}
QWidget* NodeProxy::embeddedWidget()
{
    if (widget == nullptr)
    {
        widget = new QWidget();
        QVBoxLayout* layout = new QVBoxLayout();
        widget->setLayout(layout);
        auto params = m_obj->getParams();
        for (auto param : params)
        {
            auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
            if (proxy)
            {
                auto widget_ = proxy->getParamWidget(widget);
                if (widget_)
                {
                    layout->addWidget(widget_);
                }
                m_param_proxies.emplace_back(proxy);
            }
        }
    }
    return widget;
}

void NodeProxy::onComponentAdded(aq::Algorithm* cpt)
{
    if (!widget)
    {
        widget = new QWidget();
        QVBoxLayout* layout = new QVBoxLayout();
        widget->setLayout(layout);
        auto params = m_obj->getParams();
        for (auto param : params)
        {
            auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
            if (proxy)
            {
                auto widget_ = proxy->getParamWidget(widget);
                if (widget_)
                {
                    layout->addWidget(widget_);
                }
                m_param_proxies.emplace_back(proxy);
            }
        }
    }
    auto layout = widget->layout();
    auto params = cpt->getParams();
    for (auto param : params)
    {
        auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
        if (proxy)
        {
            auto widget_ = proxy->getParamWidget(widget);
            if (widget_)
            {
                layout->addWidget(widget_);
            }
            m_param_proxies.emplace_back(proxy);
        }
    }
    if (params.size())
    {
        emit portsChanged();
    }
}
GraphProxy::GraphProxy(rcc::shared_ptr<aq::IGraph>&& ds) : m_obj(std::move(ds))
{
}
GraphProxy::GraphProxy(const rcc::shared_ptr<aq::IGraph>& ds) : m_obj(ds)
{
}
QString GraphProxy::caption() const
{
    return "Graph";
}
QString GraphProxy::name() const
{
    return "Graph";
}
std::unique_ptr<QtNodes::NodeDataModel> GraphProxy::clone() const
{
    return std::unique_ptr<GraphProxy>(new GraphProxy(std::move(aq::IGraph::create("", ""))));
}
unsigned int GraphProxy::nPorts(QtNodes::PortType portType) const
{
    if (portType == QtNodes::PortType::Out)
        return 1;
    return 0;
}
QtNodes::NodeDataType GraphProxy::dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const
{
    QtNodes::NodeDataType output;
    if (port_type == QtNodes::PortType::Out)
    {
        output.id = "aq::nodes::Node";
        output.name = "children";
    }
    return output;
}
void GraphProxy::setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port)
{
}
std::shared_ptr<QtNodes::NodeData> GraphProxy::outData(QtNodes::PortIndex port)
{
    return std::make_shared<DSOutProxy>(m_obj);
}
QWidget* GraphProxy::embeddedWidget()
{
    // TODO
    return nullptr;
}

FrameGrabberProxy::FrameGrabberProxy(rcc::shared_ptr<aq::nodes::Node>&& obj) : NodeProxy(std::move(obj))
{
    m_fg = m_obj.DynamicCast<aq::nodes::IFrameGrabber>();
}

FrameGrabberProxy::FrameGrabberProxy(const rcc::shared_ptr<aq::nodes::Node>& obj) : NodeProxy(std::move(obj))
{
    m_fg = m_obj.DynamicCast<aq::nodes::IFrameGrabber>();
}
QWidget* FrameGrabberProxy::embeddedWidget()
{
    if (widget == nullptr)
    {
        NodeProxy::embeddedWidget();
        aq::nodes::FrameGrabberInfo* info =
            dynamic_cast<aq::nodes::FrameGrabberInfo*>(m_obj->GetConstructor()->GetObjectInfo());
        selector = new QComboBox(widget);
        selector->setEditable(true);
        selector->addItem("none");
        if (info)
        {
            auto paths = info->listLoadablePaths();
            if (paths.size())
            {
                for (const auto& path : paths)
                {
                    selector->addItem(QString::fromStdString(path));
                }
            }
        }

        connect(selector->lineEdit(), SIGNAL(returnPressed()), this, SLOT(onSelectionChanged()));
        connect(selector, SIGNAL(currentIndexChanged(int)), this, SLOT(onSelectionChanged(int)));
        widget->layout()->addWidget(selector);
    }
    return widget;
}

void FrameGrabberProxy::onSelectionChanged()
{
    QString text = selector->lineEdit()->text();
    MO_LOG(info) << "Loading " << text.toStdString();
    m_fg->loadData(text.toStdString());
}

void FrameGrabberProxy::onSelectionChanged(int value)
{
    QString text = selector->currentText();
    MO_LOG(info) << "Loading " << text.toStdString();
    m_fg->loadData(text.toStdString());
}
