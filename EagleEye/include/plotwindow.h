#ifndef PLOTWINDOW_H
#define PLOTWINDOW_H
#include "Aquila/plotters/Plotter.h"
#include "qcustomplot.h"

#include <boost/shared_ptr.hpp>
#include <shared_ptr.hpp>

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
class IPlotWindow: public QWidget
{
    Q_OBJECT
public:
    explicit IPlotWindow(QWidget* parent = 0);

    virtual QString getPlotName() = 0;

public slots:
    virtual void addPlotter(rcc::shared_ptr<aq::QtPlotter> plotter) = 0;

signals:
    virtual void onDrop();
};


class PlotWindow : public IPlotWindow
{
    Q_OBJECT
    QString _name;
public:
    explicit PlotWindow(QWidget *parent = 0);
    virtual ~PlotWindow();
    virtual QString getPlotName(){return _name;}
    bool eventFilter(QObject *, QEvent *);

public slots:
    void addPlotter(rcc::shared_ptr<aq::QtPlotter> plotter);
    void on_resizePlot_activated();
signals:
    void onDrop();
private slots:

private:

    Ui::PlotWindow *ui;
    QCustomPlot* plot;
    std::vector<rcc::shared_ptr<aq::QtPlotter>> plots;
    QMenu* rightClickMenu;
};

class MatrixViewWindow: public IPlotWindow
{
    Q_OBJECT
    QString _name;
public:
    explicit MatrixViewWindow(QWidget* parent = 0);
    virtual ~MatrixViewWindow();
    void addPlotter(rcc::shared_ptr<aq::QtPlotter> plotter);
    QString getPlotName() {return _name;}
signals:
    void onDrop();
};


#endif // PLOTWINDOW_H
