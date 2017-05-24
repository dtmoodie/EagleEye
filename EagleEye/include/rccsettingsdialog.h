#ifndef RCCSETTINGSDIALOG_H
#define RCCSETTINGSDIALOG_H

#include <QDialog>
#include <MetaObject/signals/TSlot.hpp>
namespace Ui {
class RCCSettingsDialog;
}

class RCCSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RCCSettingsDialog(QWidget *parent = 0);
    ~RCCSettingsDialog();
    void updateDisplay();
private slots:
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();

    void on_comboBox_currentIndexChanged(int index);
    void on_btnAddIncludeDir_clicked();
    void on_btnAddLinkDir_clicked();
    void on_btnTestRcc_clicked();
    void on_btn_abort_compilation_clicked();

private:
    mo::TSlot<void(void)> on_constructors_added;
    Ui::RCCSettingsDialog *ui;
};

#endif // RCCSETTINGSDIALOG_H
