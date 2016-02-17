#pragma once
#include "../ParameteredObject.h"

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
            QT_Plotter = 0
        };

		virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual void Init(bool firstInit);

		virtual void SetInput(Parameters::Parameter::Ptr param_ = Parameters::Parameter::Ptr());

        virtual bool AcceptsParameter(Parameters::Parameter::Ptr param) = 0;
		virtual void OnParameterUpdate(cv::cuda::Stream* stream) = 0;
        virtual std::string PlotName() const = 0;
        virtual PlotterType Type() const = 0;

	protected:
		std::shared_ptr<Signals::connection> bc;
		Parameters::Parameter::Ptr param;
		cv::cuda::Stream plottingStream;
		cv::cuda::Event plottingEvent;
    };

	class CV_EXPORTS QtPlotter : public Plotter
    {
    protected:
        std::list<QWidget*> plot_widgets;
    public:

		virtual void AddPlot(QWidget* plot_) = 0;
		virtual void Serialize(ISimpleSerializer *pSerializer);
		virtual PlotterType Type() const;


		virtual QWidget* CreatePlot(QWidget* parent) = 0;
        virtual QWidget* GetControlWidget(QWidget* parent) = 0;
    };



}


