#ifndef SIGNAL_DIALOG_H
#define SIGNAL_DIALOG_H

#include <QDialog>

namespace mo
{
    class RelayManager;
}
class QTreeWidgetItem;

namespace Ui {
class signal_dialog;
}

class signal_dialog : public QDialog
{
    Q_OBJECT

public:
    signal_dialog(mo::RelayManager* manager, QWidget *parent = 0);
    ~signal_dialog();
    void update(mo::RelayManager* manager);
public slots:
    void on_item_select(QTreeWidgetItem* item, int column);
private:
    Ui::signal_dialog *ui;
    mo::RelayManager* _manager;
};

#endif // SIGNAL_DIALOG_H
