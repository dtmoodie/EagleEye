#include "include/bookmark_dialog.h"
#include "ui_bookmark_dialog.h"

bookmark_dialog::bookmark_dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::bookmark_dialog)
{
    ui->setupUi(this);
}

bookmark_dialog::~bookmark_dialog()
{
    delete ui;
}
