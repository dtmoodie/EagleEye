#include "csv_wizard.h"
#include "ui_csv_wizard.h"

csv_wizard::csv_wizard(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::csv_wizard)
{
    ui->setupUi(this);
}

csv_wizard::~csv_wizard()
{
    delete ui;
}


void csv_wizard::on_btn_clear_current_routine_clicked()
{

}

void csv_wizard::on_btn_accept_current_routine_clicked()
{

}

void csv_wizard::on_btn_save_current_routine_clicked()
{

}

void csv_wizard::on_btn_accept_cancel_accepted()
{

}
