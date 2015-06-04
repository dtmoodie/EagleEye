#ifndef PLOTWIZARDDIALOG_H
#define PLOTWIZARDDIALOG_H

#include <QDialog>
#include <QVector>

#include "Parameters.h"
#include "plotwindow.h"
#include "parameterplotter.h"
#include "../RuntimeObjectSystem/IObject.h"
#include "plotters/Plotter.h"
namespace Ui {
class PlotWizardDialog;
}

/*  The plot wizard dialog class will be shown whenever new data needs to be plotted.
 *  It will present the user several ways of plotting data based on the selected input
 *  It will then ask the user which plot to insert the data into.
 *  It will then spawn a ParameterPlotter object which will handle plotting and pass that object to the selected plot
*/
class QCustomPlotter;
class PlotWizardDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PlotWizardDialog(QWidget *parent = 0);
    ~PlotWizardDialog();
    bool eventFilter(QObject *, QEvent *);
signals:
    void on_plotAdded(QWidget* plot);
    void update(int idx);
public slots:
    void setup();
    void plotParameter(EagleLib::Parameter::Ptr param);
    void onUpdate(int idx);
    void handleUpdate(int idx);
private slots:
    void on_drop();
    void on_addPlotBtn_clicked();
    void on_tabWidget_currentChanged(int index);
private:

    // These are all the plot windows currently in the environment
    QVector<IPlotWindow*> plotWindows;
    // These are all the
    QVector<QWidget*> previewPlots;

    // These are all the ploters which generate data from parameters that go into the plot window
    QVector<shared_ptr<EagleLib::QtPlotter>> previewPlotters;
    QVector<shared_ptr<EagleLib::QtPlotter>> plotters;
    Ui::PlotWizardDialog *ui;
    QList<QCheckBox*> plotOptions;

    shared_ptr<EagleLib::QtPlotter> currentPlotter;
    std::vector<boost::posix_time::ptime> lastUpdateTime;
};

#endif // PLOTWIZARDDIALOG_H
