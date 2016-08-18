#pragma once
#include <EagleLib/Detail/Export.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <IObjectInfo.h>
#include <list>

// EagleLib only contains the interface for the plotting mechanisms, actual implementations will be handled inside of
// the plotting plugin

class QCustomPlot;
class QWidget;

namespace mo
{
    class IParameter;
    class Context;
}


namespace EagleLib
{
    class EAGLE_EXPORTS Plotter : public TInterface<IID_Plotter, mo::IMetaObject>
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

        virtual void SetInput(mo::IParameter* param_ = nullptr);

        virtual bool AcceptsParameter(mo::IParameter* param) = 0;
        MO_BEGIN(Plotter)
            MO_SLOT(void, on_parameter_update, mo::Context*, mo::IParameter*);
            MO_SLOT(void, on_parameter_delete, mo::IParameter const*);
        MO_END;


        virtual std::string PlotName() const = 0;
        virtual PlotterType Type() const = 0;
    protected:
        mo::IParameter* param;
    };

    struct EAGLE_EXPORTS PlotterInfo: public IObjectInfo
    {
        virtual int GetInterfaceId();
        virtual Plotter::PlotterType GetPlotType() = 0;
        virtual bool AcceptsParameter(mo::IParameter* param) = 0;
    };

    class EAGLE_EXPORTS QtPlotter : public Plotter
    {
        class impl;
        std::shared_ptr<impl> _pimpl;
    protected:
        std::list<QWidget*> plot_widgets;
    public:
        virtual mo::IParameter* addParameter(mo::IParameter* param);
        virtual mo::IParameter* addParameter(std::shared_ptr<mo::IParameter> param);
        virtual void AddPlot(QWidget* plot_) = 0;
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual PlotterType Type() const;

        virtual QWidget* CreatePlot(QWidget* parent) = 0;
        virtual QWidget* GetControlWidget(QWidget* parent);
    };
}


