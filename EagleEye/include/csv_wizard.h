#ifndef CSV_WIZARD_H
#define CSV_WIZARD_H

#include <QDialog>

namespace Ui {
class csv_wizard;
}

class csv_wizard : public QDialog
{
    Q_OBJECT

public:
    explicit csv_wizard(QWidget *parent = 0);
    ~csv_wizard();

private slots:

    void on_btn_clear_current_routine_clicked();

    void on_btn_accept_current_routine_clicked();

    void on_btn_save_current_routine_clicked();

    void on_btn_accept_cancel_accepted();

private:
    Ui::csv_wizard *ui;
};

#endif // CSV_WIZARD_H
