#pragma once
#include "EagleLib//ParameteredIObject.h"

#include "parameters/Parameters.hpp"

#include "RuntimeLinkLibrary.h"
#include "ObjectInterface.h"
#include "ObjectInterfacePerModule.h"
#include "IObject.h"

#include <opencv2/core/cuda.hpp>

// EagleLib only contains the interface for the plotting mechanisms, actual implementations will be handled inside of
// the plotting plugin

class QCustomPlot;
class QWidget;


namespace EagleLib
{
    
    class CV_EXPORTS Plotter : public TInterface<IID_Plotter, ParameteredIObject>
    {
    public:
        Plotter();
        virtual ~Plotter();
        enum PlotterType
        {
            QT_Plotter = 1 << 0,
            VTK_Plotter = 1 << 1
        };

        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual void Init(bool firstInit);
        virtual void PlotInit(bool firstInit);

        virtual void SetInput(Parameters::Parameter* param_ = nullptr);

        virtual bool AcceptsParameter(Parameters::Parameter* param) = 0;
        virtual void OnParameterUpdate(cv::cuda::Stream* stream) = 0;
        virtual std::string PlotName() const = 0;
        virtual PlotterType Type() const = 0;
        virtual void on_param_delete(Parameters::Parameter* param);
    protected:
        Parameters::Parameter* param;
        cv::cuda::Stream plottingStream;
        cv::cuda::Event plottingEvent;
    };

    struct EAGLE_EXPORTS PlotterInfo: public IObjectInfo
    {
        virtual int GetObjectInfoType();
        virtual Plotter::PlotterType GetPlotType() = 0;
        virtual bool AcceptsParameter(Parameters::Parameter* param) = 0;
    };

    class CV_EXPORTS QtPlotter : public Plotter
    {
        class impl;
        std::shared_ptr<impl> _pimpl;
    protected:
        std::list<QWidget*> plot_widgets;
    public:
        virtual Parameters::Parameter* addParameter(Parameters::Parameter* param);
        virtual Parameters::Parameter* addParameter(std::shared_ptr<Parameters::Parameter> param);
        virtual void AddPlot(QWidget* plot_) = 0;
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual PlotterType Type() const;

        virtual QWidget* CreatePlot(QWidget* parent) = 0;
        virtual QWidget* GetControlWidget(QWidget* parent);
    };



}


