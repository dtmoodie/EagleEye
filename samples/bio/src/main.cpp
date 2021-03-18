#include <aqcore/display/ImageDisplay.h>

#include <Aquila/core.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/gui.hpp>
#include <Aquila/gui/ViewControllers.hpp>

#include <MetaObject/object/RelayManager.hpp>

#include <aqcore/imgproc/Blur.hpp>
#include <aqcore/imgproc/Channels.h>
#include <aqcore/imgproc/HoughCircles.hpp>
#include <aqframegrabbers/directory.h>
#include <bio/MeasureCell.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("image_directory", boost::program_options::value<std::string>(), "Path with images");
    boost::program_options::variables_map vm;
    auto parsed_options =
        boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    boost::program_options::store(parsed_options, vm);

    if (vm.count("image_directory") == 0)
    {
        std::cout << "Expect --image_directory argument to point to a directory of images for processing" << std::endl;
        std::cout << "Example usage: bio_example --image_directory=C:\\data\\2-14-18" << std::endl;
        boost::this_thread::sleep_for(boost::chrono::seconds(5));
        return 1;
    }

    std::string file_path = vm["image_directory"].as<std::string>();
    if (!boost::filesystem::exists(file_path) || !boost::filesystem::is_directory(file_path))
    {
        std::cout << file_path << " not a valid directory" << std::endl;
        boost::this_thread::sleep_for(boost::chrono::seconds(5));
        return 1;
    }

    std::cout << "Loading " << file_path << std::endl;

    std::shared_ptr<SystemTable> table = SystemTable::instance();
    mo::MetaObjectFactory::Ptr_t factory = mo::MetaObjectFactory::instance(table.get());
    factory->registerTranslationUnit();

    aq::core::initModule(factory.get());
    aq::gui::initModule(factory.get());
    aqcore::initPlugin(factory.get());
    aqbio::initPlugin(factory.get());
    aqframegrabbers::initPlugin(factory.get());

    std::cout << "Initialization complete" << std::endl;
    auto gui_thread = aq::gui::createGuiThread();
    auto graph = aq::IGraph::create();
    graph->stop();
    auto fg = aqframegrabbers::FrameGrabberDirectory::create();
    fg->synchronous = true;
    auto kbcontroller = aq::gui::KeyboardSignalController::create();
    kbcontroller->signal_map_param.setValue({{13, "nextFrame"}});
    kbcontroller->print_unused_keys = true;

    kbcontroller->setupSignals(graph->getRelayManager());
    graph->addNode(fg);
    fg->loadData(file_path);

    auto gray = aqcore::ConvertToGrey::create();
    gray->connectInput(*fg, "output", "input");

    auto blur = aqcore::GaussianBlur::create();
    blur->connectInput(*gray, "grey", "input");
    blur->sigma = 2.0;

    auto circles = aqcore::HoughCircle::create();
    circles->connectInput(*blur, "output", "input");
    circles->center_threshold = 12;

    auto membrane = aqbio::FindCellMembrane::create();
    membrane->connectInput(*blur, "output", "input");
    membrane->connectInput(*circles, "circles", "circles");
    membrane->alpha = 2.0;
    membrane->beta = 2.0;
    membrane->num_samples = 400;
    membrane->window_size = 9;
    membrane->inner_pad = 1.1f;
    membrane->outer_pad = 2.5f;
    membrane->radial_resolution = 0.5;
    membrane->radial_weight = 4.5;

    auto measure = aqbio::MeasureCell::create();
    measure->connectInput(*membrane, "", "cell");
    measure->connectInput(*fg, "", "image_name");
    measure->connectInput(*fg, "", "image");
    measure->out_dir = file_path + "/results";

    auto disp = aq::nodes::QtImageDisplay::create();
    disp->connectInput(*measure, "overlay", "image");

    std::cout << "Starting to process data" << std::endl;

    graph->start();
    graph->waitForSignal("eos");
    std::cout << "Done!!!!!" << std::endl;
    return 0;
}
