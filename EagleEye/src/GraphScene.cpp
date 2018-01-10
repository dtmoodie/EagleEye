#include "GraphScene.hpp"
#include "Aquila/core/IGraph.hpp"
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/serialization/cereal/JsonArchive.hpp"
#include "MetaObject/serialization/cereal_map.hpp"
#include "Node.hpp"
#include "object_proxies.hpp"
#include "object_proxies.hpp"
#include <boost/asio.hpp>
#include <qfiledialog.h>

#include <boost/log/attributes.hpp>
#include <fstream>

namespace cereal
{
    template <class AR, class T, int rows, int cols>
    void serialize(AR& ar, cv::Matx<T, rows, cols>& mat)
    {
        cereal::size_type size = rows * cols;
        ;
        ar(cereal::make_size_tag(size));
        MO_ASSERT(size == rows * cols);
        for (int i = 0; i < rows * cols; ++i)
            ar(mat.val[i]);
    }
}

GraphScene::GraphScene(std::shared_ptr<QtNodes::DataModelRegistry> registry) : FlowScene(registry)
{
}

void GraphScene::save(bool)
{
    auto selected_nodes = selectedNodes();
    for (auto node : selected_nodes)
    {
        auto proxy = dynamic_cast<aq::GraphProxy*>(node->nodeDataModel());
        if (proxy)
        {
            QString filename = QFileDialog::getSaveFileName(
                nullptr, tr("Open Flow Scene"), QDir::homePath(), tr("Flow Scene Files (*.json)"));

            std::ofstream ofs(filename.toStdString());
            aq::JSONOutputArchive ar(ofs, aq::JSONOutputArchive::Options(), *vm, *sm);
            auto ds_nodes = proxy->m_obj->getAllNodes();
            std::map<std::string, cv::Vec2f> node_position_map;

            for (auto node_itr = this->_nodes.begin(); node_itr != this->_nodes.end(); ++node_itr)
            {
                if (auto node_proxy = dynamic_cast<aq::NodeProxy*>(node_itr->second->nodeDataModel()))
                {
                    for (const auto& node : ds_nodes)
                    {
                        if (node_proxy->m_obj == node)
                        {
                            QtNodes::NodeGraphicsObject& ngo = node_itr->second->nodeGraphicsObject();
                            node_position_map[node->getTreeName()] = cv::Vec2f(ngo.pos().x(), ngo.pos().y());
                            continue;
                        }
                    }
                    continue;
                }
                if (auto fg_proxy = dynamic_cast<aq::FrameGrabberProxy*>(node_itr->second->nodeDataModel()))
                {
                    for (const auto& node : ds_nodes)
                    {
                        if (fg_proxy->m_obj == node)
                        {
                            QtNodes::NodeGraphicsObject& ngo = node_itr->second->nodeGraphicsObject();
                            node_position_map[node->getTreeName()] = cv::Vec2f(ngo.pos().x(), ngo.pos().y());
                            continue;
                        }
                    }
                    continue;
                }
            }
            ar(cereal::make_nvp("ui_node_positions", node_position_map));
            std::vector<rcc::shared_ptr<aq::IGraph>> dsvec;
            dsvec.push_back(proxy->m_obj);
            aq::IGraph::save(ar, dsvec);

            return;
        }
    }
}

void GraphScene::load()
{
    clearScene();

    QString fileName =
        QFileDialog::getOpenFileName(nullptr, tr("Open Flow Scene"), QDir::homePath(), tr("Flow Scene Files (*.json)"));

    if (!QFileInfo::exists(fileName))
        return;
    std::ifstream ifs(fileName.toStdString());
    std::map<std::string, std::string> _vm, _sm;
    std::map<std::string, std::string>& vm = this->vm ? *this->vm : _vm;
    std::map<std::string, std::string>& sm = this->sm ? *this->sm : _sm;

    aq::JSONInputArchive ar(ifs, vm, sm);
    aq::IGraph::VariableMap defaultVM, defaultSM;

    std::stringstream currentDate;
    boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
    currentDate << timeLocal.date().year() << "-" << std::setfill('0') << std::setw(2)
                << timeLocal.date().month().as_number() << "-" << std::setfill('0') << std::setw(2)
                << timeLocal.date().day().as_number();
    sm["${date}"] = currentDate.str();
    sm["${hostname}"] = boost::asio::ip::host_name();
    currentDate.str(std::string());
    currentDate << std::setfill('0') << std::setw(2) << timeLocal.time_of_day().hours() << std::setfill('0')
                << std::setw(2) << timeLocal.time_of_day().minutes();
    sm["${hour}"] = currentDate.str();
    sm["${pid}"] = boost::lexical_cast<std::string>(boost::log::aux::this_process::get_id());
    sm["${config_file_dir}"] = boost::filesystem::path(fileName.toStdString()).parent_path().string();

    ar(cereal::make_optional_nvp("DefaultVariables", defaultVM, defaultVM));
    ar(cereal::make_optional_nvp("DefaultStrings", defaultSM, defaultSM));
    for (const auto& pair : defaultVM)
    {
        if (vm.count(pair.first) == 0)
            vm[pair.first] = pair.second;
    }
    for (const auto& pair : defaultSM)
    {
        if (sm.count("${" + pair.first + "}") == 0)
            sm["${" + pair.first + "}"] = pair.second;
    }
    if (sm.size())
    {
        std::stringstream ss;
        for (const auto& pair : sm)
            ss << "\n" << pair.first << " = " << pair.second;
        MO_LOG(debug) << "Used string replacements: " << ss.str();
    }

    if (vm.size())
    {
        std::stringstream ss;
        for (const auto& pair : vm)
            ss << "\n" << pair.first << " = " << pair.second;
        MO_LOG(debug) << "Used variable replacements: " << ss.str();
    }
    auto dsvec = aq::IGraph::load(ar);
    for (auto& ds : dsvec)
    {
        load(ds);
    }
    std::map<std::string, cv::Vec2f> node_position_map;
    ar(cereal::make_optional_nvp("ui_node_positions", node_position_map, node_position_map));
    for (auto node_itr = this->_nodes.begin(); node_itr != this->_nodes.end(); ++node_itr)
    {
        if (auto node_proxy = dynamic_cast<aq::NodeProxy*>(node_itr->second->nodeDataModel()))
        {
            auto itr = node_position_map.find(node_proxy->m_obj->getTreeName());
            if (itr != node_position_map.end())
            {
                node_itr->second->nodeGraphicsObject().setPos(itr->second.val[0], itr->second.val[1]);
            }
        }
        if (auto fg_proxy = dynamic_cast<aq::FrameGrabberProxy*>(node_itr->second->nodeDataModel()))
        {
            auto itr = node_position_map.find(fg_proxy->m_obj->getTreeName());
            if (itr != node_position_map.end())
            {
                node_itr->second->nodeGraphicsObject().setPos(itr->second.val[0], itr->second.val[1]);
            }
        }
    }
    for (auto& connection : this->_connections)
    {
        connection.second->getConnectionGraphicsObject().move();
    }
}

void GraphScene::loadFromMemory(const QByteArray& data)
{
}

QtNodes::Node& GraphScene::load(const rcc::shared_ptr<aq::IGraph>& ds)
{
    rcc::shared_ptr<aq::IGraph> ds_ = ds;
    std::unique_ptr<aq::GraphProxy> datamodel(new aq::GraphProxy(ds));
    auto node = std::make_unique<QtNodes::Node>(std::move(datamodel));
    auto ngo = std::make_unique<QtNodes::NodeGraphicsObject>(*this, *node);
    node->setGraphicsObject(std::move(ngo));
    auto nodePtr = node.get();
    this->_nodes[node->id()] = std::move(node);

    nodeCreated(*nodePtr);
    std::map<std::string, QtNodes::Node*> nodemap;
    auto nodes = ds_->getTopLevelNodes();
    for (const auto& node : nodes)
    {
        QtNodes::Node& created = load(node, nodemap);
        createConnection(created, 0, *nodePtr, 0);
    }
    // Make a pass through all the nodes and reconstruct input / output connections
    for (const auto& node : nodes)
    {
        reconnectInputs(node, nodemap);
    }
    return *nodePtr;
}

QtNodes::Node& GraphScene::load(const rcc::shared_ptr<aq::nodes::INode>& node,
                                std::map<std::string, QtNodes::Node*>& nodemap)
{
    auto itr = nodemap.find(node->getTreeName());
    if (itr != nodemap.end())
    {
        return *itr->second;
    }
    rcc::shared_ptr<aq::nodes::Node> node_ = node;
    std::unique_ptr<aq::NodeProxy> datamodel;
    if (node_->GetInterface(aq::nodes::IFrameGrabber::s_interfaceID))
    {
        datamodel = std::make_unique<aq::FrameGrabberProxy>(node_);
    }
    else
    {
        datamodel = std::make_unique<aq::NodeProxy>(node_);
    }
    auto node_obj = std::make_unique<QtNodes::Node>(std::move(datamodel));
    auto ngo = std::make_unique<QtNodes::NodeGraphicsObject>(*this, *node_obj);
    node_obj->setGraphicsObject(std::move(ngo));
    auto nodePtr = node_obj.get();
    this->_nodes[node_obj->id()] = std::move(node_obj);

    nodeCreated(*nodePtr);
    auto nodes = node_->getChildren();
    for (const auto& node : nodes)
    {
        QtNodes::Node& created = load(node, nodemap);
        createConnection(created, 0, *nodePtr, 0);
    }
    nodemap[node->getTreeName()] = nodePtr;
    return *nodePtr;
}
void GraphScene::reconnectInputs(const rcc::shared_ptr<aq::nodes::INode>& node,
                                 std::map<std::string, QtNodes::Node*>& nodemap)
{
    auto inputs = node->getInputs();
    auto input_node_itr = nodemap.find(node->getTreeName());
    if (input_node_itr != nodemap.end())
    {
        for (int i = 0; i < inputs.size(); ++i)
        {
            auto input_param = inputs[i]->getInputParam();
            if (input_param)
            {
                auto output_name = input_param->getTreeRoot();
                auto itr = nodemap.find(output_name);
                if (itr != nodemap.end())
                {
                    int nout = itr->second->nodeDataModel()->nPorts(QtNodes::PortType::Out);
                    for (int j = 0; j < nout; ++j)
                    {
                        auto dtype = itr->second->nodeDataModel()->dataType(QtNodes::PortType::Out, j);
                        if (dtype.name.toStdString() == input_param->getName())
                        {
                            createConnection(*input_node_itr->second, i + 1, *itr->second, j);
                            continue;
                        }
                    }
                }
            }
        }
    }
    auto children = node->getChildren();
    for (auto child : children)
    {
        reconnectInputs(child, nodemap);
    }
}
