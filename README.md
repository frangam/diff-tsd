# Execution in background 
Also, you can run the script with [nohup](https://en.wikipedia.org/wiki/Nohup) which ignores the hangup signal. This means that you can close the terminal without stopping the execution. Also, don’t forget to add & so the script runs in the background:

```sh
$ nohup accelerate launch train.py --config CONFIG_FILE  > your.log &
```

In addition, to close remote terminal safely, you have to run the exit commant, and do not close manually the terminal:

```sh
$ exit
```

Finally, you can identify the running processes of training script with:
```sh
$ ps ax | grep train.py
```

or list of all running processes of Python:
```sh
$ ps -fA | grep python
```

then, kill the desired one:

```sh
$ kill PID
```

# Create Data Splits
In all bash command, we can combine the use "nohup" command to execute a script withouth interuptions (avoiding terminal disconnections, etc.) and "&" symbol at the end of the command for a background execution. We also can use "> filename.log" to put the results in a log file.

## Sampling techniques
- The Leave-One-Trial-Out (LOTO) approach is a cutting-edge method in sample generation. Each trial encompasses a unique raw activity signal for a single subject, ensuring an impartial evaluation and facilitating the creation of a sufficient number of samples. Additionally, this technique prevents the duplication of trials with identical raw signals (trials of the same label) across both training and testing datasets.
- The Leave-One-Subject-Out (LOSO) approach is a sampling technique inspired by the Leave-One-Trial-Out method. In this approach, all trials belonging to a single subject are considered as an indivisible unit, ensuring that there are no trials from the same subject duplicated in the training and testing datasets. This technique maintains data integrity and prevents potential biases caused by the presence of trials from the same subject in both datasets, allowing for a more robust and reliable evaluation of the model's performance. This technique is the most strict, which proposes a subject-wise approach instead record-wise, and in the literature is not commonly assessed, maybe due to its resulting lower accuracy.

## Create Recurrence Plots
- "--create-numpies" is for create the first time the numpy arrays.
- With "--sampling loto" you can select the sampling method "loto" or "loso" with "--sampling loso".

For LOTO approach:
```sh
 $ nohup ./generate_recurrence_plots.py --create-numpies --data-name WISDM --n-folds 3 --data-folder YOUR_DATA_PATH --sampling loto > recurrence_plots_loto.log &
```


### Uploading to HuggingFace
Then, we have to upload the recurrence plots to Huggingface platform:

```sh
 $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loto --huggingface-token YOUR_TOKEN > upload_rp_loto.log &
```

# Training
First, activate your Python VENV where you have all dependencies installed:

```sh
$ source ~/YOUR_VENV/bin/activate
```

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

on all nodes, then running:

For LOTO approach:

```sh
$ nohup accelerate launch ./train.py --config configs/config_wisdm_128x128_loto.json --max-epochs 10000 --batch-size 16 > train_loto.log &
```
For LOSO approach:

```sh
$ nohup accelerate launch ./train.py --config configs/config_wisdm_128x128_loso.json --max-epochs 10000 --batch-size 16 > train_loso.log &
```

on all nodes.

# Sampling

Generate "n" samples (in this case, 2.000).

```sh
$  nohup ./sample.py --config configs/config_wisdm_128x128_loto.json -n 2000  > sample-loto.log &
```

```sh
$  nohup ./sample.py --config configs/config_wisdm_128x128_loso.json -n 2000  > sample-loso.log &
```

# Prepare to evaluate the shyntetic samples generated
Create splits of images for train/test and assess the quality of synthetic images generated before.

- Synthetic images are the train set.
- TEST set are real data, which are used to validate the model in the training phase Test set are real data, which is used to test the trained model.
- This also  copy the real images used to train the diffusion models to the folder result (you could use this real images to asses the quality of real-based recurrence plots to recognize activities).


```sh
$ nohup ./tgen/data.py --config configs/config_wisdm_128x128_loto.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > data_splits-loto.log &
```

```sh
$ nohup ./tgen/data.py --config configs/config_wisdm_128x128_loso.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > data_splits-loso.log &
```

# Evaluation 
## Evaluation of synthetic recurrence plots sampled
We can evaluate the synthetic sampled images in the recognition of activities. We can use a set of benchmarking [image classifiers](https://github.com/qubvel/classification_models).


Example of evaluation of synthtetic (using LOTO approach and xception model)
```sh
$  nohup ./eval_diffusion.py --model-name xception --prefix exp-classes-all-classes --epochs 100 --synth-train --config configs/config_wisdm_128x128_loto.json > eval-synth-loto-xception.log &
```

## Evaluation of real recurrence plots to compare with synthetic
```sh
$  nohup ./eval_diffusion.py --model-name xception --prefix exp-classes-3-4 --epochs 100 --synth-train  --config configs/config_wisdm_128x128_loto.json > eval-real-loto-xception.log &
```


---

## Clone a repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

Now that you're more familiar with your Bitbucket repository, go ahead and add a new file locally. You can [push your change back to Bitbucket with SourceTree](https://confluence.atlassian.com/x/iqyBMg), or you can [add, commit,](https://confluence.atlassian.com/x/8QhODQ) and [push from the command line](https://confluence.atlassian.com/x/NQ0zDQ).