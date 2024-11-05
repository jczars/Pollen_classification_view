## reports_build
import sys, gc
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)
from models import reports

def reports_build(conf, test_data_generator, model, CATEGORIES, tempo, return_0):
    save_dir  = return_0['save_dir_train']
    batch_size=conf['batch_size']
    nm_model  =conf['model']

    (test_loss, test_accuracy) = model.evaluate(test_data_generator,
                                                batch_size=batch_size, verbose=1)

    y_true, y_pred, df_cor=reports.predict_data_generator(conf, return_0,
                                                          test_data_generator,
                                                          model,
                                                          tempo,
                                                          CATEGORIES)

    reports.plot_confusion_matrix(y_true, y_pred, CATEGORIES, nm_model,
                              save_dir, tempo)

    #-------Rel_class
    reports.class_reports(y_true, y_pred, CATEGORIES, nm_model, save_dir,
                      tempo)

    me=reports.metricas(y_true, y_pred)

    me={'test_loss':test_loss,
         'test_accuracy':test_accuracy,
         'precision':me['precision'],
         'recall':me['recall'],
         'fscore':me['fscore'],
         'kappa':me['kappa'],
         }
    reports.boxplot(nm_model, df_cor, save_dir, tempo)
    return me

if __name__=="__main__": 
    help(reports_build)