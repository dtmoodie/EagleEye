#pragma once

#include "Parameters.h"


#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "../../RuntimeObjectSystem/ObjectInterface.h"
#include "../../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "../../RuntimeObjectSystem/IObject.h"

// EagleLib only contains the interface for the plotting mechanisms, actual implementations will be handled inside of
// the plotting plugin






class QCustomPlot;
class QWidget;


namespace EagleLib
{

    class Plotter: public TInterface<IID_Plotter, IObject>
    {

    protected:
        boost::signals2::connection bc;
        EagleLib::Parameter::Ptr param;
        boost::function<void(void)> f;
    public:
        enum PlotterType
        {
            QT_Plotter = 0
        };
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            SERIALIZE(bc);
            SERIALIZE(param);
        }

        virtual void setInput(EagleLib::Parameter::Ptr param_ = EagleLib::Parameter::Ptr())
        {
            bc.disconnect();
            param = param_;
            if(param)
                bc = param->onUpdate.connect(boost::bind(&Plotter::onUpdate, this));
        }

        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const = 0;
        virtual void doUpdate() = 0;

        virtual void onUpdate()
        {
            if(f) f();
        }
        virtual std::string plotName() const = 0;
        virtual PlotterType type() const = 0;
        /**
         * @brief setCallback sets a callback that is called inside the UI code
         * @brief this can be used for example with QT to call a function to signal that the
         * @brief UI needs to be updated on the main thread's event loop
         * @param f_ is the function that is called from the thread that updates a parameter
         */

        virtual void setCallback(boost::function<void(void)> f_)
            {f = f_;}
    };

    // The workflow for Qt visualization objects is as follows:
    // 1) Check if a parameter can be plotted by a certain plotter via the acceptsType parameter
    // 2) A parameter is set to monitor via setInput
    // 3) A plot is set via the addPlot method
    // 3)

    class QtPlotter: public Plotter
    {
    protected:
        std::vector<QWidget*> plots;
    public:

        virtual QWidget* getPlot();

        virtual void addPlot(QWidget* plot_)
        {
            plots.push_back(plot_);
        }
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            Plotter::Serialize(pSerializer);
            SERIALIZE(plots);
        }
        /**
         * @brief acceptsWidget determines if this plotter can be dropped into a particular widget
         * @param widget widget is the end point of the drop
         * @return true if it can go into that widget, false otehrwise
         */
        virtual bool acceptsWidget(QWidget* widget) = 0;

        /**
         * @brief acceptsType
         * @param param
         * @return
         */
        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const = 0;



        /**
         * @brief plotName is the name of this specific plotting implementation
         * @return
         */
        virtual std::string plotName() const = 0;

        /**
         * @brief type returns the type of plotter
         */
        virtual PlotterType type() const
        {
            return QT_Plotter;
        }

        virtual QWidget* getSettingsWidget() const = 0;
    };
//    class TestPlot: public QtPlotter
//    {
//    public:
//        TestPlot();
//        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const;

//        virtual std::string plotName() const;

//        virtual QWidget* getSettingsWidget() const;
//    };


}


