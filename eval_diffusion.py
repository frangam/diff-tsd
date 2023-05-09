#!/home/fmgarmor/miot_env/bin/python3

import os
import numpy as np
import pandas as pd
from functools import partial
import argparse
from tqdm import trange, tqdm
from torchvision import datasets, transforms, utils
from datasets import load_dataset
import keras
import k_diffusion as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input


from tgen import eval




def main():
    '''Examples of runs:
    - train with synthetic images
    $  nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-all-classes --epochs 100 --synth-train --config configs/config_wisdm_128x128_loso.json > results/evaluation_synthetic_quality/exp-all-classes/eval.log &

    $  nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-classes-3-4 --epochs 100 --synth-train > results/evaluation_synthetic_quality/exp-classes-3-4/eval.log &
    
    - train with real images
    $  nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-classes-3-4 --epochs 100 > results/evaluation_synthetic_quality/exp-classes-3-4/eval.log &


    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-name', type=str, default="lstm-cnn", #vgg16
                   help='the model name') #models: https://github.com/qubvel/classification_models    
                   #"resnet18", "resnet50", "densenet121", "mobilenet", "simple-cnn"
    p.add_argument('--synth-train', action="store_true",
                   help='use synthetic images for train; if not, use real images') 
    p.add_argument('--prefered-device', type=int, default=-1, help='the prefered device to run the model')    
    p.add_argument('--img-batch-size', type=int, default=64,
                   help='the batch size for loading the images')         
    p.add_argument('--batch-size', type=int, default=256,
                   help='the batch size for the neural network')
    p.add_argument('--image-size', type=int, default=32,
                   help='the image resolution size')
    p.add_argument('--n-folds', type=int, default=3,
                   help='the number of folds')
    p.add_argument('--prefix', type=str, default='exp-all-classes',
                   help='the output prefix')
    p.add_argument('--epochs', type=int, default=100,
                   help='the number of epochs')
    p.add_argument('--val-split', type=float, default=0.0,
                   help='if we use a validation split, indicate a value; if not, 0.0')
    p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",help='the configuration file')

    args = p.parse_args()
    
    config = K.config.load_config(open(args.config))
    dataset_config = config['dataset']
    sampling_method = dataset_config["sampling"]

    use_synth_images_for_train = args.synth_train
    print("use synthetic images for train?: ", use_synth_images_for_train)
    val_split = args.val_split
    model_name = args.model_name
    img_size = args.image_size
    epochs = args.epochs
    batch_size = args.batch_size
    img_batch_size = args.img_batch_size
    prefered_device = args.prefered_device
    # config = K.config.load_config(open(args.config))
    # model_config = config['model']
    # dataset_config = config['dataset']
    # size = model_config['input_size']
    # tf = transforms.Compose([
    #     transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
    #     transforms.CenterCrop(size[0]),
    #     K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob']),
    # ])

    # test_set = load_dataset(f"{dataset_config['location']}{args.current_fold}", split="test")
    # test_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
    # print(test_set)

    # CLASS_NAMES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing"]
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []


    models, histories = [], []
    acc_per_class = {}

    dir_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/result_models/" 
    os.makedirs(dir_path, exist_ok=True)
    # release GPU Memory
    # K.clear_session()
    if val_split > 0:
        train_idg = ImageDataGenerator(rescale=1./255,  validation_split=val_split)
    else:
        train_idg = ImageDataGenerator(rescale=1./255)

    val_idg = ImageDataGenerator(rescale=1./255)
    test_idg = ImageDataGenerator(rescale=1./255)


    for fold in range(args.n_folds):
        train_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/"
        train_path = f"{train_path}train/" if use_synth_images_for_train else f"{train_path}real_train/"
        test_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/test/"
        val_path = test_path #f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/validation/"
        train_data = pd.read_csv(f'{train_path}{"" if use_synth_images_for_train else "real_"}training_labels.csv', dtype=str)[["filename", "label"]]
        test_data = pd.read_csv(f'{test_path}test_labels.csv', dtype=str)[["filename", "label"]]
        val_data = test_data#pd.read_csv(f'{val_path}val_labels.csv', dtype=str)[["filename", "label"]]
        # print("train path: ", train_path)
        # print("train shape: ", train_data.shape)
        # print("val path: ", val_path)
        # print("val shape: ", val_data.shape)
        # print("test path: ", test_path)
        # print("test shape: ", test_data.shape)

        print("Loading images")
        # from tgen.preprocess_crop import preprocess_crop
        #training data are the synthetic images
        
        
        if val_split > 0:
            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        subset='training',
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
        # for training - we used a 20% of train data to validate the model during training phase             
            valid_data_generator  = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        subset='validation',
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
        else:
            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
            valid_data_generator  = val_idg.flow_from_dataframe(val_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = val_path,
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )

        #to test the trained model, we used real test data unseen on training phase
        test_data_generator  = test_idg.flow_from_dataframe(test_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = test_path,
                    x_col = "filename", y_col = "label",
                    # class_mode = "raw", 
                    class_mode="categorical",
                    shuffle = False, #important shuffle=False to compare the prediction result
                    ) 
        # print("train_data_generator.class_indices", train_data_generator.class_indices)
        # print("train_data_generator.labels", train_data_generator.labels)
        # print("valid_data_generator.class_indices", valid_data_generator.class_indices)
        # print("valid_data_generator.labels", valid_data_generator.labels)
        # print("test_data_generator.class_indices", test_data_generator.class_indices)
        # print("test_data_generator.labels", test_data_generator.labels)
        
        # STEP_SIZE_TRAIN=train_data_generator.n // train_data_generator.batch_size
        # STEP_SIZE_VALID=valid_data_generator.n // valid_data_generator.batch_size
        STEP_SIZE_TEST=test_data_generator.n // test_data_generator.batch_size

        train_labels = (train_data_generator.class_indices)
        train_labels = dict((v,k) for k,v in train_labels.items())
        n_classes =  len(train_labels)

        print("Class indices:", train_labels)
        print("Number of classes:", n_classes)
        print("Building model:", model_name)

        #Train the model
        if model_name == "simple":
            model, history = eval.build_model_A(dir_path, model_name, fold, n_classes, (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device) #train2_ds, val2_ds, EPOCHS)
        elif model_name == "lstm-cnn":
            model, history = eval.generate_lstmfcn_xia(dir_path, model_name, fold, n_classes, (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device)
        else:
            model, history = eval.build_X(dir_path, model_name, fold, n_classes,  (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device) #train2_ds, val2_ds, EPOCHS)
        
        print("model built")
        models.append(model)
        histories.append(history)

        #plot history
        eval.plot_train_eval(history, f"{p}% Synthetic Images") 

        # LOAD BEST MODEL to evaluate the performance of the model
        # model.load_weights(f"{dir_path}{model_name}/model_{fold}")
        
        # results = model.evaluate(valid_data_generator)
        results = model.evaluate(valid_data_generator, steps=STEP_SIZE_TEST)
        results = dict(zip(model.metrics_names,results))
        print("Results - validation:", results)
        
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])


        #confusion matrix of current fold
        # y_pred = model.predict(test_data_generator)
        test_data_generator.reset() #You need to reset the test_generator before whenever you call the predict_generator. This is important, if you forget to reset the test_generator you will get outputs in a weird order.
        y_pred=model.predict(test_data_generator, steps=STEP_SIZE_TEST, verbose=1)
        # print("y_pred:", y_pred)
        predicted_class_indices = np.argmax(y_pred,axis=1)
        # print("predicted_class_indices:", predicted_class_indices)
        # predicted_class_indices_axis__1 = np.argmax(y_pred,axis=-1)
        # print("predicted_class_indices_axis_-1:", predicted_class_indices_axis__1)

        #now predicted_class_indices has the predicted labels, but you can’t simply tell what the predictions are, because all you can see is numbers like 0,1,4,1,0,6…
        #and most importantly you need to map the predicted labels with their unique ids such as filenames to find out what you predicted for which image.
        
        # print("original labels:", labels)
        test_labels = (test_data_generator.class_indices)
        test_labels = dict((v,k) for k,v in test_labels.items())
        predictions = [test_labels[k] for k in predicted_class_indices]
        # print("test labels true:", test_data_generator.labels)
        # print("predictions:", predictions)


        # Finally, save the results to a CSV file.
        print("Saving predictions")
        filenames=test_data_generator.filenames
        print("filenames:", len(filenames))
        print("predictions:", len(predictions))
        results=pd.DataFrame({"Filename": filenames, "Predictions": predictions})
        os.makedirs(f"{dir_path}fold_{fold}/{model_name}/", exist_ok=True)
        results.to_csv(f"{dir_path}fold_{fold}/{model_name}/prediction_results.csv",index=False)

        # y_pred = np.argmax(y_pred, axis=1) #-1) 
        metrics = eval.plot_cm(
            f'{dir_path}fold_{fold}/{model_name}/',
            model_name,
            test_data_generator.labels,
            predicted_class_indices,
            train_labels.values(),
            normalized=True,
            figsize=(12, 8),
            suffix_name=f"_fold_{fold}"
            )
        
        # laux = list(labels.values())
        # for j,d in enumerate(diag):
        #     if laux[j] not in acc_per_class:
        #         acc_per_class[laux[j]] = []
        #     acc_per_class[laux[j]].append(d)
        
        # acc_per_class[f"Confussion Matrix metrics [fold-{fold}]"] = metrics
        acc_per_class[f"Average Accuracy [fold-{fold}]"] = [VALIDATION_ACCURACY[fold]]
        acc_per_class[f"Average Loss [fold-{fold}]"] = [VALIDATION_LOSS[fold]]

        # tf.keras.backend.clear_session()
    acc_per_class[f"Average Accuracy"] = [np.mean(np.array(VALIDATION_ACCURACY))]
    acc_per_class[f"Average Loss"] = [np.mean(np.array(VALIDATION_LOSS))]

    print(acc_per_class)
    
    # av_cols = [f"fold-{f}" for f in range(args.n_folds)]
    # av_cols.append("mean")
    avg_accuracies_df = pd.DataFrame(acc_per_class)
    os.makedirs(f"{dir_path}average/{model_name}/", exist_ok=True)
    avg_accuracies_df.to_csv(f"{dir_path}average/{model_name}/test_accuracies.csv",index=False)

    #plot only one, since all of them share the same architecture
    keras.utils.plot_model(model, f"{dir_path}average/{model_name}/model_multi_input_and_output_model.png", show_shapes=True)


if __name__ == '__main__':
    main()