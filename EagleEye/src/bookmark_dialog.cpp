#include "bookmark_dialog.h"
#include "ui_bookmark_dialog.h"

bookmark_dialog::bookmark_dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::bookmark_dialog)
{
    ui->setupUi(this);
    connect(ui->list_bookmarks, SIGNAL(itemDoubleClickd(QListWidgetItem*)), this, SLOT(on_file_selected(QListWidgetItem*)));
    connect(ui->list_history, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this,SLOT(on_file_selected(QListWidgetItem*)));
}

bookmark_dialog::~bookmark_dialog()
{
    delete ui;
}

void bookmark_dialog::update()
{

}

void bookmark_dialog::append_history(std::string dir)
{

}
void bookmark_dialog::on_file_selected(QListWidgetItem* item)
{
    std::string name = item->text().toStdString();
    if(sender() == ui->list_bookmarks)
    {
        
    }
    if(sender() == ui->list_history)
    {
        
    }
    emit ope_file(name);
}