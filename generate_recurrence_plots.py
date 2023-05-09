#!/home/fmgarmor/miot_env/bin/python3

from keras.utils import to_categorical
import argparse
import tgen.activity_data as act
import tgen.recurrence_plots as rec
def main():
    '''Examples of runs:
    $ ./generate_recurrence_plots.py
    $ nohup ./generate_recurrence_plots.py --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/ --sampling loso --create-numpies > recurrence_plots_loso.log &

    $ nohup ./generate_recurrence_plots.py --create-numpies --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/ --sampling loto > recurrence_plots_loto.log &
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-name', type=str, default="WISDM", help='the database name')
    p.add_argument('--data-folder', type=str, default="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", help='the data folder path')
    p.add_argument('--n-folds', type=int, default=3, help='the number of k-folds')
    p.add_argument('--sampling', type=str, default="loso", help='loso: leave-one-subject-out; loto: leave-one-trial-out')
    p.add_argument('--create-numpies', action="store_true", help='create numpies before; if not, load numpies')


    args = p.parse_args()
    create_numpies = args.create_numpies
    data_folder = args.data_folder
    data_name = args.data_name
    FOLDS_N = args.n_folds
    TIME_STEPS, STEPS = act.get_time_setup(DATASET_NAME=data_name)
    X_train, y_train, sj_train = None, None, None
    if not create_numpies:
        print("Loading numpies...")
        X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False, sampling=args.sampling)
    else:
        print("Creating numpies...")
        X_train, y_train, sj_train = act.create_all_numpy_datasets(data_name, data_folder, COL_SELECTED_IDXS=list(range(3, 3+3)), sampling=args.sampling)
        y_train = to_categorical(y_train, dtype='uint8') 
    print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
    
    rec.generate_all_recurrence_plots(X_train, y_train, sj_train, data_folder, TIME_STEPS, FOLDS_N, args.sampling)


if __name__ == '__main__':
    main()