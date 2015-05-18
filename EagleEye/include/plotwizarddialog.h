#ifndef PLOTWIZARDDIALOG_H
#define PLOTWIZARDDIALOG_H

#include <QDialog>

namespace Ui {
class PlotWizardDialog;
}

/*  The plot wizard dialog class will be shown whenever new data needs to be plotted.
 *  It will present the user several ways of plotting data based on the selected input
 *  It will then ask the user which plot to insert the data into.
 *  It will then spawn a ParameterPlotter object which will handle plotting and pass that object to the selected plot
*/

class PlotWizardDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PlotWizardDialog(QWidget *parent = 0);
    ~PlotWizardDialog();

private:
    Ui::PlotWizardDialog *ui;
};

#endif // PLOTWIZARDDIALOG_H
