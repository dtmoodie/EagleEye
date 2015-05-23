#include "plotters/Plotter.h"

namespace EagleLib
{
    class HistogramPlotter: public QtPlotter
    {
    public:

        HistogramPlotter();
        virtual bool acceptsType(EagleLib::Parameter::Ptr param) const;
        virtual std::string plotName() const;
        virtual QWidget* getSettingsWidget() const;
    };
}
