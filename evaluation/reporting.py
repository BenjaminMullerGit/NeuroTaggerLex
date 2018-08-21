import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_curves(error_train,error_dev, error_test,epochs, model, data_set, epoch_max, target_path,
                info_suff="",
                loss_train=None, loss_dev=None, loss_test=None, save=True):
    if info_suff != "":
        info_suff = "-"+info_suff
    dir_ = os.path.join(target_path, model)
    if loss_train is not None:
        assert loss_dev is not None and loss_test is not None
        plt.figure()
        plt.plot(epochs, loss_train, 'c--',loss_dev, 'r--', epochs, loss_test, 'b--')
        plt.xlabel("num epochs")
        plt.ylabel("LOSS")
        plt.title("Learning curves model {} on data {} epoch max {}".format(model,data_set,epoch_max))
        plt.show()
        if save:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)
                print("INFO Reports : dir {} created ".format(os.path.join(target_path,model)))
            plt.savefig(os.path.join(dir_, model+"-"+data_set+"-losses"+info_suff))

    plt.figure()
    plt.plot(epochs, error_train, "c--", epochs, error_dev, 'r--', epochs, error_test, 'b--')
    plt.xlabel("num epochs")
    plt.ylabel("Accuracy UPOS")
    plt.title("Learning curves model {} on data {} epoch max {}".format(model,data_set,epoch_max))
    if save:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
            print("INFO Reports : dir {} created ".format(os.path.join(target_path,model)))
        plt.savefig(os.path.join(dir_,model+"-"+data_set+"-accuracy"+info_suff))

if __name__=="__main__":
    plot_curves([0.1,0.3], [0.2, 0.3],[1,2],loss_train=[0.5,0.3], loss_dev=[0.23,0.12],loss_test=[0.2,0.1],model="MOffDEL",data_set= "DAvTA2",epoch_max=2,target_path="../")