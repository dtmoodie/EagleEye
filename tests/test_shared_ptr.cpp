


int main()
{
    /*{
        rcc::shared_ptr<aq::Node> nullPtr;
        {
            rcc::shared_ptr<aq::Node> ptr(aq::NodeManager::getInstance().addNode("VideoLoader"));
            rcc::shared_ptr<aq::Plotter> plotPtr = aq::PlotManager::getInstance().getPlot("TestPlot");
            rcc::shared_ptr<aq::QtPlotter> qPlotPtr(plotPtr);
            for(int i = 0; i < 10; ++i)
            {
                aq::NodeManager::getInstance().CheckRecompile(true);
                std::cout << "Node: " << ptr.get() << " Notifiers: ";
                {
                    rcc::shared_ptr<aq::Node> cpyConstructor(ptr);
                    cpyConstructor->drawResults = false;
                    for(size_t i = 0; i < ptr->notifiers.size(); ++i)
                    {
                        std::cout << ptr->notifiers[i] << " ";
                    }
                    std::cout << std::endl;
                }
                {
                    rcc::shared_ptr<aq::Node> cpy(ptr);
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
