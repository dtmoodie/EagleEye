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

        virtual bool setInput(EagleLib::Parameter::Ptr param) = 0;
        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const = 0;
        virtual void onUpdate() const
        { if(f) f();}
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


    class QtPlotter: public Plotter
    {

        std::vector<QCustomPlot*> plots;
    public:
        virtual void addPlot(QCustomPlot* plot_)
        {
            plots.push_back(plot_);
        }
        virtual void Serialize(ISimpleSerializer *pSerializer)
        {
            Plotter::Serialize(pSerializer);
            SERIALIZE(plots);
        }

        virtual bool setInput(EagleLib::Parameter::Ptr param_)
        {
            bc.disconnect();
            param = param_;
            bc = param_->onUpdate.connect(boost::bind(&QtPlotter::onUpdate, this));
        }

        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const = 0;


        /**
         * @brief onUpdate is called by a parameter's signal whenever a parameter is updated
         */
        virtual void onUpdate() const
        {Plotter::onUpdate(); }

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

}


