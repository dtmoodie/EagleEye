#include <Aquila/core.hpp>
#include <Aquila/core.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/gui.hpp>
#include <Aquila/gui/ViewControllers.hpp>

#include <MetaObject/object/RelayManager.hpp>
#include <aqcore/display/ImageDisplay.h>
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

    SystemTable table;
    mo::MetaObjectFactory factory(&table);
    factory.registerTranslationUnit();

    aq::core::initModule(&factory);
    aq::gui::initModule(&factory);
    aqcore::initPlugin(&factory);
    aqbio::initPlugin(&factory);
    aqframegrabbers::initPlugin(&factory);

    std::cout << "Initialization complete" << std::endl;
    auto gui_thread = aq::gui::createGuiThread();
    auto graph = aq::IGraph::create();
    graph->stopThread();
    auto fg = aq::nodes::FrameGrabberDirectory::create();
    fg->synchronous = true;
    auto kbcontroller = aq::gui::KeyboardSignalController::create();
    kbcontroller->signal_map_param.updateData({ 
        {13, "nextFrame"}
    });
    kbcontroller->print_unused_keys = true;
    
    kbcontroller->setupSignals(graph->getRelayManager());
    graph->addNode(fg);
    fg->loadData(file_path);

    auto gray = aq::nodes::ConvertToGrey::create();
    gray->connectInput(fg, "output", "input");

    auto blur = aq::nodes::GaussianBlur::create();
    blur->connectInput(gray, "grey", "input");
    blur->sigma = 2.0;

    auto circles = aq::nodes::HoughCircle::create();
    circles->connectInput(blur, "output", "input");
    circles->center_threshold = 12;

    auto membrane = aq::bio::FindCellMembrane::create();
    membrane->connectInput(blur, "output", "input");
    membrane->connectInput(circles, "circles", "circles");
    membrane->alpha = 2.0;
    membrane->beta = 2.0;
    membrane->num_samples = 400;
    membrane->window_size = 9;
    membrane->inner_pad = 1.1f;
    membrane->outer_pad = 2.5f;
    membrane->radial_resolution = 0.5;
    membrane->radial_weight = 4.5;

    auto measure = aq::bio::MeasureCell::create();
    measure->connectInput(membrane, "", "cell");
    measure->connectInput(fg, "", "image_name");
    measure->connectInput(fg, "", "image");
    measure->out_dir = file_path + "/results";

    auto disp = aq::nodes::QtImageDisplay::create();
    disp->connectInput(measure, "overlay", "image");

    std::cout << "Starting to process data" << std::endl;

    graph->startThread();
    graph->waitForSignal("eos");
    std::cout << "Done!!!!!" << std::endl;
    return 0;
}
