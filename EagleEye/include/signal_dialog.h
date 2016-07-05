#ifndef SIGNAL_DIALOG_H
#define SIGNAL_DIALOG_H

#include <QDialog>

namespace Signals
{
    class signal_manager;
}
class QTreeWidgetItem;

namespace Ui {
class signal_dialog;
}

class signal_dialog : public QDialog
{
    Q_OBJECT

public:
    signal_dialog(Signals::signal_manager* manager, QWidget *parent = 0);
    ~signal_dialog();
    void update(Signals::signal_manager* manager);
public slots:
    void on_item_select(QTreeWidgetItem* item, int column);
private:
    Ui::signal_dialog *ui;
    Signals::signal_manager* _manager;
};

#endif // SIGNAL_DIALOG_H
