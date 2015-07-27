#ifndef RCCSETTINGSDIALOG_H
#define RCCSETTINGSDIALOG_H

#include <QDialog>

namespace Ui {
class RCCSettingsDialog;
}

class RCCSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RCCSettingsDialog(QWidget *parent = 0);
    ~RCCSettingsDialog();

private slots:
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();

    void on_comboBox_currentIndexChanged(int index);
    void on_btnAddIncludeDir_clicked();
    void on_btnAddLinkDir_clicked();

private:
    Ui::RCCSettingsDialog *ui;
};

#endif // RCCSETTINGSDIALOG_H
