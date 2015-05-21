#include "plotters/Plotter.h"
#include "qcustomplot.h"

namespace EagleLib
{
    class HistogramPlotter: public QtPlotter
    {

    public:
        HistogramPlotter();

    };
    REGISTERCLASS(HistogramPlotter)
}
