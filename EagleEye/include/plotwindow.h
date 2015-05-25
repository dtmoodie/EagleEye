#ifndef PLOTWINDOW_H
#define PLOTWINDOW_H
#include "qcustomplot.h"
#include "plotters/Plotter.h"
#include <boost/shared_ptr.hpp>

/* The plot window class used used to create a widget that looks as follows
 *
 *   --------------------------------------------------
 *   |                                     (plot name1)|
 *   |                                     (plot name2)|
 *   |                                     (plot name3)|
 *   |   [Plot Area]                       (plot name4)|
 *   |                                                 |
 *   |                                  [Plot settings]|
 *   --------------------------------------------------
 *
 *   The plot names section will contain the shorthand name of the variable being plotted.
 *   Hovering over plot names will give the full variable path
 *   Plot settings will change parameters such as sliding window size
 *   Clicking each individual plot will spawn a dialog to change individual plot parameters
 *
*/
#include <QWidget>

namespace Ui {
class PlotWindow;
}

class PlotWindow : public QWidget
{
    Q_OBJECT
    QString _name;
public:
    explicit PlotWindow(QWidget *parent = 0);
    ~PlotWindow();
    QString getPlotName(){return _name;}
    bool eventFilter(QObject *, QEvent *);
public slots:
    void addPlotter(shared_ptr<EagleLib::QtPlotter> plotter);
signals:
    void onDrop();
private:
    Ui::PlotWindow *ui;
    QCustomPlot* plot;
    std::vector<shared_ptr<EagleLib::QtPlotter>> plots;
};

#endif // PLOTWINDOW_H
