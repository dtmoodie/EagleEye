#pragma once
#include "FlowScene.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
namespace aq{
    class IDataStream;
    namespace Nodes{
        class Node;
    }
}
class GraphScene: public QtNodes::FlowScene{
    Q_OBJECT
public:
    GraphScene(std::shared_ptr<QtNodes::DataModelRegistry> registry =
                std::make_shared<QtNodes::DataModelRegistry>());
public slots:
    void save(bool);
    void load();
public:
    virtual void loadFromMemory(const QByteArray& data);
    virtual QtNodes::Node& load(const rcc::shared_ptr<aq::IDataStream>& ds);
    virtual QtNodes::Node& load(const rcc::shared_ptr<aq::Nodes::Node>& node, std::map<std::string, QtNodes::Node*>& nodemap);
    virtual void reconnectInputs(const rcc::shared_ptr<aq::Nodes::Node>& node, std::map<std::string, QtNodes::Node*>& nodemap);
    void setVmSm(std::map<std::string, std::string>* vm, std::map<std::string, std::string>* sm) {
        this->sm = sm; this->vm = vm;
    }
protected:
    std::map<std::string, std::string>* sm = nullptr;
    std::map<std::string, std::string>* vm = nullptr;
};