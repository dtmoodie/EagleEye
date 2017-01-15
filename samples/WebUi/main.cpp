#include <EagleLib/IDataStream.hpp>
#include <EagleLib/Nodes/Node.h>
#include <EagleLib/Logging.h>
#include <EagleLib/Nodes/IFrameGrabber.hpp>

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

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <functional>
using namespace EagleLib;
using namespace EagleLib::Nodes;
using namespace mo;
using namespace Wt;
struct GlobalContext
{
    std::vector<rcc::shared_ptr<IDataStream>> _data_streams;

    TypedSignal<void(void)> onStreamAdded;
    TypedSignal<void(IDataStream*, Nodes::Node*)> onNodeAdded;
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
          _btn_add_node->clicked().connect(std::bind(&MainApplication::onActionClicked, this, _btn_add_node));
          
          _btn_load_data = new WPushButton(_action_list_container);
          _btn_load_data->setText("Load data");
          _btn_load_data->clicked().connect(std::bind(&MainApplication::onActionClicked, this, _btn_load_data));

          _btn_load_config = new WPushButton(_action_list_container);
          _btn_load_config->setText("Load config file");
          _btn_load_config->clicked().connect(std::bind(&MainApplication::onActionClicked, this, _btn_load_config));


        //_graph = new WTree();
        //_graph->setSelectionMode(SingleSelection);

    }
    
    void onAddNodeClicked()
    {
        
    }
    
protected:

    void onStreamAdded()
    {
    
    }

    void onNodeAdded(IDataStream* stream, Nodes::Node*)
    {
        
    }
    void onActionClicked(WPushButton* sender)
    {
        if(sender == _btn_load_data)
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
                    for(auto& document : documents)
                    {
                        table->elementAt(count, 0)->addWidget(new WText(document));
                        table->elementAt(count, 1)->addWidget(new WText(fg_info->GetDisplayName()));
                        ++count;
                    }
                }
            }

            WLineEdit* manual_entry = new WLineEdit("Enter file path", dialog->contents());


            WPushButton* btn_ok = new WPushButton("OK", dialog->footer());
            btn_ok->setDefault(true);
            btn_ok->clicked().connect(std::bind([=]() {
                dialog->accept();
            }));

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
    
    // for now displaying the graph as a tree
    Wt::WTree* _graph;

    rcc::weak_ptr<IDataStream> _current_stream;
    rcc::weak_ptr<Nodes::Node> _current_node;
    IParameter* _current_parameter;

    WContainerWidget* _data_stream_list_container;
    WContainerWidget* _action_list_container;
      Wt::WPushButton* _btn_add_node;
      Wt::WPushButton* _btn_load_data;
      Wt::WPushButton* _btn_load_config;
      Wt::WPushButton* _btn_run_script;
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
    
    boost::filesystem::path currentDir = boost::filesystem::path(argv[0]).parent_path();
#ifdef _MSC_VER
    currentDir = boost::filesystem::path(currentDir.string());
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
#endif
    LOG(info) << "Looking for plugins in: " << currentDir.string();

    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(currentDir))
    {
        for (boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
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
