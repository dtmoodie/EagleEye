#include "MetaObject/Parameters/UI/Wt/POD.hpp"
#include "MetaObject/Parameters/UI/Wt/IParameterProxy.hpp"
#include <EagleLib/IDataStream.hpp>
#include <EagleLib/Nodes/Node.h>
#include <EagleLib/Logging.h>
#include <EagleLib/Nodes/IFrameGrabber.hpp>
#include <EagleLib/Nodes/NodeFactory.h>

#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Demangle.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/UI/WT.hpp>
#include <MetaObject/Parameters/UI/Wt/IParameterInputProxy.hpp>
#include <MetaObject/Parameters/UI/Wt/IParameterProxy.hpp>
#include <MetaObject/Parameters/UI/Wt/IParameterOutputProxy.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Detail/Allocator.hpp>
#include <MetaObject/Thread/ThreadPool.hpp>
#include <MetaObject/Parameters/UI/Wt/IParameterInputProxy.hpp>
#include <MetaObject/Parameters/UI/Wt/IParameterOutputProxy.hpp>


#include "instantiate.hpp"

#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>
#include <Wt/WSlider>
#include <Wt/WSpinBox>
#include <Wt/WComboBox>
#include <Wt/WTree>
#include <Wt/WTable>
#include <Wt/WTreeNode>
#include <Wt/WDialog>
#include <Wt/WSuggestionPopup>
#include <Wt/WSortFilterProxyModel>

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <functional>
using namespace EagleLib;
using namespace EagleLib::Nodes;
using namespace mo;
using namespace Wt;

INSTANTIATE_META_PARAMETER(bool);
INSTANTIATE_META_PARAMETER(int);
INSTANTIATE_META_PARAMETER(unsigned short);
INSTANTIATE_META_PARAMETER(unsigned int);
INSTANTIATE_META_PARAMETER(char);
INSTANTIATE_META_PARAMETER(unsigned char);
INSTANTIATE_META_PARAMETER(float);
INSTANTIATE_META_PARAMETER(double);
INSTANTIATE_META_PARAMETER(std::string);

struct GlobalContext
{
    std::vector<rcc::shared_ptr<IDataStream>> _data_streams;

    TypedSignal<void(void)> onStreamAdded;
    TypedSignal<void(IDataStream*, Nodes::Node*)> onNodeAdded;
    boost::filesystem::path _current_dir;
};
GlobalContext g_ctx;



class MainApplication: public UI::wt::MainApplication
{
public:
    MainApplication(const Wt::WEnvironment& env):
        UI::wt::MainApplication(env)
    {
        _action_list_container = new WContainerWidget(root());
          _btn_add_node = new WPushButton(_action_list_container);
          _btn_add_node->setText("Add node");
          _btn_add_node->clicked().connect(std::bind(&MainApplication::onAddNodeClicked, this));
          
          _btn_load_data = new WPushButton(_action_list_container);
          _btn_load_data->setText("Load data");
          _btn_load_data->clicked().connect(std::bind(&MainApplication::onLoadDataClicked, this));

          _btn_load_config = new WPushButton(_action_list_container);
          _btn_load_config->setText("Load config file");
          _btn_load_config->clicked().connect(std::bind(&MainApplication::onLoadConfigClicked, this));

          _btn_plugins = new WPushButton(_action_list_container);
          _btn_plugins->setText("List loaded plugins");


        _data_stream_list_container = new WContainerWidget(root());

        _node_container = new WContainerWidget(root());
          _node_container->setWidth(WLength("20%"));
          _node_tree = new WTree(_node_container);
          _node_tree->setSelectionMode(SingleSelection);
          _parameter_display = new WContainerWidget(_node_container);
          _parameter_display->hide();
    }

    
protected:

    void onStreamAdded()
    {

    }

    void onNodeAdded(IDataStream* stream, Nodes::Node*)
    {

    }

    void onLoadConfigClicked()
    {

    }

    void onAddNodeClicked()
    {
        if(_current_stream || _current_node)
        {
            WDialog* dialog = new WDialog("Select node", this);

            auto constructors = MetaObjectFactory::Instance()->GetConstructors(Nodes::Node::s_interfaceID);

            WTable* table = new WTable(dialog->contents());
            table->setHeaderCount(1);
            table->elementAt(0, 0)->addWidget(new WText("Name"));
            table->elementAt(0, 1)->addWidget(new WText("Description"));
            table->setWidth(WLength("100%"));
            int column = 0;
            int row = 1;
            for(int i = 0; i < constructors.size(); ++i)
            {
                IObjectConstructor* constructor = constructors[i];
                std::string node_name = constructor->GetName();
                Nodes::NodeInfo* info = dynamic_cast<Nodes::NodeInfo*>(constructors[i]->GetObjectInfo());
                if(info)
                {
                    WPushButton* name = new WPushButton(info->GetDisplayName());
                    table->elementAt(row, column * 2)->addWidget(name);
                    WPushButton* btn = new WPushButton();
                    btn->setText("Detailed info");
                    table->elementAt(row, column * 2 + 1)->addWidget(btn);
                    name->clicked().connect(
                        std::bind([node_name, this]()
                        {
                            if(_current_stream && !_current_node)
                            {
                                _current_stream->AddNode(node_name);
                            }
                            if(_current_node && _current_stream)
                            {
                                auto added_nodes = NodeFactory::Instance()->AddNode(node_name, _current_node.Get());
                                if (added_nodes.size() == 0)
                                    return;
                                WTreeNode* root = _node_tree->treeRoot();
                                for(auto& node : _display_nodes)
                                {
                                    root->removeChildNode(node.second);
                                }
                                _display_nodes.clear();
                                std::vector<rcc::weak_ptr<Nodes::Node>> nodes = _current_stream->GetTopLevelNodes();
                                std::string new_node_name = added_nodes[0]->GetTreeName();
                                for(auto& node : nodes)
                                {
                                    if(auto ret = populateTree(node, root, new_node_name))
                                    {
                                        // When you add a node, select it in the tree
                                        //_node_tree->select(ret);
                                    }
                                }
                                this->_current_node = added_nodes[0];
                            }
                        }));
                    ++row;
                    if(row > 15)
                    {
                        row = 1;
                        ++column;
                    }
                }
            }
            WPushButton* btn_cancel = new WPushButton("Cancel", dialog->footer());
            btn_cancel->clicked().connect(dialog, &Wt::WDialog::reject);

            dialog->rejectWhenEscapePressed();
            dialog->setModal(true);

            dialog->finished().connect(std::bind([=]()
            {
                delete dialog;
            }));

            dialog->show();
        }
    }

    void onLoadDataClicked()
    {
        WDialog* dialog = new WDialog("Select data to load", this);

        auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IFrameGrabber::s_interfaceID);
        std::vector<std::pair<std::string, std::string>> data;

        WTable *table = new Wt::WTable(dialog->contents());
          table->setHeaderCount(1);
          table->elementAt(0, 0)->addWidget(new WText("File"));
          table->elementAt(0, 0)->addWidget(new WText("FrameGrabber"));
          table->setMargin(5);

          int count = 1;
          for (auto constructor : constructors)
          {
              auto fg_info = dynamic_cast< IFrameGrabber::InterfaceInfo*>(constructor->GetObjectInfo());
              if (fg_info)
              {
                  auto documents = fg_info->ListLoadableDocuments();
                  std::string fg_name = fg_info->GetDisplayName();
                  for(auto document : documents)
                  {
                      table->elementAt(count, 0)->addWidget(new WText(document));
                      table->elementAt(count, 1)->addWidget(new WText(fg_name));

                      WPushButton* btn = new WPushButton();
                      btn->setText("Load");
                      table->elementAt(count, 2)->addWidget(btn);
                      btn->clicked().connect(std::bind([document, fg_name, dialog, this]()
                      {
                          this->loadData(document, fg_name);
                          dialog->accept();
                      }));
                      ++count;
                  }
              }
          }

        WLineEdit* manual_entry = new WLineEdit(dialog->contents());
        manual_entry->setEmptyText("Enter file path");

        manual_entry->enterPressed().connect(
            std::bind([manual_entry, dialog, this]()
            {
                std::string data = manual_entry->text().toUTF8();
                this->loadData(data);
                dialog->accept();
            }));
        WSuggestionPopup::Options options;
        options.highlightBeginTag = "<span class=\"highlight\">";
        options.highlightEndTag = "</span>";
        //options.listSeparator = ',';
        options.whitespace = " \\n";
        options.wordSeparators = "-., \"@\\n;";
        //options.appendReplacedText = ", ";
        _sp = new WSuggestionPopup(options, dialog->contents());

        _sp->filterModel().connect(this, &MainApplication::onFilterModel);

        boost::filesystem::directory_iterator end_itr;
        _current_path = g_ctx._current_dir;
        if (boost::filesystem::is_directory(g_ctx._current_dir))
        {
            for (boost::filesystem::directory_iterator itr(g_ctx._current_dir); itr != end_itr; ++itr)
            {
                if (boost::filesystem::is_regular_file(itr->path()))
                {
                    std::string file = itr->path().stem().string();

                    _sp->addSuggestion(file);
                }
            }
        }
#ifdef _MSC_VER

#else
        for(boost::filesystem::directory_iterator itr("/"); itr != end_itr; ++itr)
        {
            _sp->addSuggestion(itr->path().string());
        }
#endif
        _sp->forEdit(manual_entry);
        _sp->setFilterLength(-1);

        WPushButton* btn_ok = new WPushButton("OK", dialog->footer());
          btn_ok->setDefault(true);
          btn_ok->clicked().connect(std::bind([=]()
          {
              std::string data = manual_entry->text().toUTF8();
              auto pos = data.find(',');
              if(pos != std::string::npos)
              {
                    data = data.substr(0, pos);
              }
              this->loadData(data);
              dialog->accept();
          }));

        WPushButton* btn_cancel = new WPushButton("Cancel", dialog->footer());
        btn_cancel->clicked().connect(dialog, &Wt::WDialog::reject);

        dialog->rejectWhenEscapePressed();
        dialog->setModal(true);

        dialog->finished().connect(std::bind([=]()
        {
            delete dialog;
            this->_sp  = nullptr;
        }));

        dialog->show();
    }
    
    void onFilterModel(const WString& data_)
    {
        std::cout << "on filter model " << data_ << std::endl;
        boost::filesystem::directory_iterator end_itr;
        std::string data = data_.toUTF8();
        for (boost::filesystem::directory_iterator itr(_current_path); itr != end_itr; ++itr)
        {
            if(data == itr->path().string())
            {
                if(boost::filesystem::is_directory(itr->path()))
                {
                    _sp->clearSuggestions();
                    for (boost::filesystem::directory_iterator itr2(itr->path()); itr2 != end_itr; ++itr2)
                    {
                        _sp->addSuggestion(itr2->path().string());
                    }
                    return;
                }
            }
        }
#ifndef _MSC_VER
        for(boost::filesystem::directory_iterator itr("/"); itr != end_itr; ++itr)
        {
            if(data == itr->path().string())
            {
                for (boost::filesystem::directory_iterator itr2(itr->path()); itr2 != end_itr; ++itr2)
                {
                    _sp->addSuggestion(itr2->path().string());
                }
                _current_path = data + "/";
                return;
            }
        }
#endif

    }

    void loadData(const std::string& data, const std::string& fg = "")
    {
        auto ds = IDataStream::Create(data, fg);
        if (ds)
        {
            ds->StartThread();
            g_ctx._data_streams.push_back(ds);
        }
        std::string display_name;
        if(boost::filesystem::is_regular_file(data))
        {
            boost::filesystem::path path(data);
            display_name = path.stem().string();
        }else
        {
            display_name = data;
        }
        rcc::weak_ptr<IDataStream> weak_ptr(ds);
        WPushButton* btn = new WPushButton(display_name, _data_stream_list_container);
        btn->clicked().connect(std::bind([weak_ptr, this, display_name]()
        {
            this->onStreamSelected(weak_ptr, display_name);
        }));
    }
    
    WTreeNode* populateTree(rcc::weak_ptr<Nodes::Node> current_node, WTreeNode* display_node, const std::string& desired_return = "")
    {
        WTreeNode* new_node = new WTreeNode(current_node->GetTreeName(), 0, display_node);
        WTreeNode* output = nullptr;
        if(current_node->GetTreeName() == desired_return)
            output = new_node;
        _display_nodes[current_node->GetTreeName()] = new_node;
        new_node->selected().connect(
            std::bind([this, current_node]()
            {
                this->onNodeSelected(current_node);

            }));
        auto children = current_node->GetChildren();
        for(auto& child : children)
        {
            auto ret = populateTree(child, new_node, desired_return);
            if(ret && ! output)
                output = ret;
        }
        return output;
    }

    void onStreamSelected(rcc::weak_ptr<IDataStream> stream, const std::string& display = "")
    {
        _current_stream = stream;
        _current_node.reset();
        _current_parameter = nullptr;
        delete _node_tree->treeRoot();
        _display_nodes.clear();
        
        WTreeNode* root_node = new WTreeNode(display);
        _node_tree->setTreeRoot(root_node);
        root_node->label()->setTextFormat(Wt::PlainText);

        std::vector<rcc::weak_ptr<Nodes::Node>> nodes = stream->GetTopLevelNodes();
        
        for(auto& node : nodes)
        {
            populateTree(node, root_node);
        }
    }

    void onNodeSelected(rcc::weak_ptr<Nodes::Node> node)
    {
        this->_current_node = node;
        _parameter_display->show();
        _parameter_display->clear();
        auto params = node->GetAllParameters();
        for(auto param : params)
        {
            auto widget = UI::wt::WidgetFactory::Instance()->CreateWidget(param, this);
            if(widget)
            {
                _parameter_display->addWidget(widget);
                if(UI::wt::WidgetFactory::Instance()->CanPlot(param))
                {
                    Wt::WPushButton* btn = new Wt::WPushButton(widget);
                    btn->setText("Plot");
                    btn->setCheckable(true);
                    btn->setChecked(false);
                    btn->clicked().connect(std::bind([widget, param, this, btn]()
                    {
                        if(btn->isChecked())
                        {
                            auto plot = UI::wt::WidgetFactory::Instance()->CreatePlot(param,this, widget);
                            if(plot)
                            {

                            }
                            btn->setText("Remove plot");
                        }else
                        {
                            auto children = widget->children();
                            for(auto child : children)
                            {
                                if(dynamic_cast<UI::wt::IPlotProxy*>(child))
                                {
                                    delete child;
                                    btn->setText("Plot");
                                    return;
                                }
                            }
                        }

                    }));
                }
            }else
            {
                if(param->CheckFlags(Input_e))
                {
                    // Get a list of all possible input parameters
                    auto box = new WComboBox();
                    box->addItem(param->GetTreeName());
                    auto all_nodes = node->GetDataStream()->GetAllNodes();
                    auto input = dynamic_cast<InputParameter*>(param);
                    if(input)
                    {
                        for(auto output_node : all_nodes)
                        {
                            auto output_params = output_node->GetAllParameters();
                            for(auto output : output_params)
                            {
                                if(input->AcceptsInput(output))
                                {
                                    box->addItem(output->GetTreeName());
                                }
                            }
                        }
                    }
                    box->changed().connect(
                        std::bind([node, box, input]
                    {
                        rcc::shared_ptr<Nodes::Node> node_ = node;
                        auto output_name = box->currentText().toUTF8();
                        if(output_name == input->GetTreeName())
                        {
                            input->SetInput(nullptr);
                            return;
                        }
                        auto pos = output_name.find(':');
                        if(pos == std::string::npos)
                            return;
                        auto output_node = node_->GetDataStream()->GetNode(output_name.substr(0, pos));
                        if(output_node)
                        {
                            auto output_param = output_node->GetOutput(output_name.substr(pos + 1));
                            if(output_param)
                            {
                                input->SetInput(output_param);
                            }
                        }
                    }));
                    _parameter_display->addWidget(box);
                }else if(param->CheckFlags(Output_e))
                {

                }else
                {
                    auto text = new WText(param->GetTreeName());
                    text->setToolTip(param->GetTypeInfo().name());
                    _parameter_display->addWidget(text);
                }

            }
        }
        _parameter_display->show();
    }

    // for now displaying the graph as a tree
    Wt::WTree* _node_tree;

    rcc::weak_ptr<IDataStream> _current_stream;
    rcc::weak_ptr<Nodes::Node> _current_node;
    IParameter* _current_parameter;

    WContainerWidget* _data_stream_list_container;
    WContainerWidget* _action_list_container;
      WPushButton* _btn_add_node;
      WPushButton* _btn_load_data;
      WPushButton* _btn_load_config;
      WPushButton* _btn_run_script;
      WPushButton* _btn_plugins;
    WContainerWidget* _node_container;
      WContainerWidget* _parameter_display;
    boost::filesystem::path _current_path;
    WSuggestionPopup *_sp = nullptr;
    std::map<std::string, WTreeNode*> _display_nodes;
};

WApplication* createApplication(const WEnvironment& env)
{
    return new MainApplication(env);
}

int main(int argc, char** argv)
{
    mo::instantiations::initialize();
    EagleLib::SetupLogging();
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
    auto g_allocator = mo::Allocator::GetThreadSafeAllocator();
    g_allocator->SetName("Global Allocator");
    mo::SetGpuAllocatorHelper<cv::cuda::GpuMat>(g_allocator);
    mo::SetCpuAllocatorHelper<cv::Mat>(g_allocator);

    boost::filesystem::path current_dir = boost::filesystem::path(argv[0]).parent_path();
    g_ctx._current_dir  = current_dir;
#ifdef _MSC_VER
    current_dir = boost::filesystem::path(current_dir.string());
#else
    current_dir = boost::filesystem::path(current_dir.string() + "/Plugins");
#endif
    LOG(info) << "Looking for plugins in: " << current_dir.string();

    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(current_dir))
    {
        for (boost::filesystem::directory_iterator itr(current_dir); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
#ifdef _MSC_VER
                if (itr->path().extension() == ".dll")
#else
                if (itr->path().extension() == ".so")
#endif
                {
                    std::string file = itr->path().string();
                    mo::MetaObjectFactory::Instance()->LoadPlugin(file);
                }
            }
        }
    }

    WRun(argc, argv, &createApplication);
}
