


int main()
{
    /*{
        rcc::shared_ptr<EagleLib::Node> nullPtr;
        {
            rcc::shared_ptr<EagleLib::Node> ptr(EagleLib::NodeManager::getInstance().addNode("VideoLoader"));
            rcc::shared_ptr<EagleLib::Plotter> plotPtr = EagleLib::PlotManager::getInstance().getPlot("TestPlot");
            rcc::shared_ptr<EagleLib::QtPlotter> qPlotPtr(plotPtr);
            for(int i = 0; i < 10; ++i)
            {
                EagleLib::NodeManager::getInstance().CheckRecompile(true);
                std::cout << "Node: " << ptr.get() << " Notifiers: ";
                {
                    rcc::shared_ptr<EagleLib::Node> cpyConstructor(ptr);
                    cpyConstructor->drawResults = false;
                    for(size_t i = 0; i < ptr->notifiers.size(); ++i)
                    {
                        std::cout << ptr->notifiers[i] << " ";
                    }
                    std::cout << std::endl;
                }
                {
                    rcc::shared_ptr<EagleLib::Node> cpy(ptr);
                    cpy->enabled = false;
                }
                boost::this_thread::sleep_for(boost::chrono::milliseconds(300));
            }
            nullPtr = ptr;
            nullPtr->enabled;
        }
    }*/
    return 0;
}
