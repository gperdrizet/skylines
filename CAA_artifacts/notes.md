# Notes

## 1. Thoughts

### 1.1

This type of project is hard to experiment with - each new tweak requires a large amount of both disk space and time:

```text
$ du -sh data/*

14G     ./gan_output
400M    ./image_datasets
1.8G    ./specimens
5.0T    ./training_checkpoints
```

That's from just one short-ish run, with one set of configuration variables and was generated over the course of ~4 days.

### 1.2

Each model is a more or less 'good' method to find city-ness in an n 100 vector of random numbers.

### 1.3

The generated cities have a concept of nearness or similarity which is not physical or aesthetic, e.g. city [1.1, 1.2,...] is closer to city [1.2, 1.2,...] than it is to city [1.8, 1.2,...].

### 1.4

The model translates 100 random numbers into 3.2 million numbers which resemble a city when formatted as a jpeg image.

## 2. Training scratch

### 2.1. 2024-02-11 run

The learning rates were initially set at 0.000025. At 7000 batches the learning rates were manually updated to 0.00001. The learning rates were updated again at around 7800 batches to 0.0001 because the models had failed to progress visually. Prior archived runs used 0.0001 with good results.

The models were trained to ~16k batches with hardly any progress. Seems like switching to a fast learning rate after the model had already started to converge was not helpful. In the future, try starting with a large learning rate and then decreasing it as training progresses.

Generated the training video out to about 9700 frames for archival purposes, but didn't keep any other artifacts.

### 2.2. 2024-02-17 run

By about batch 18000 it was apparent that the model was flopping around - it was still generating some interesting results, but not really making progress. It would get better and then worse again on the scale of about 100 batches. So at 19000 batches, training was stopped, the learning rates were adjusted from 0.00005 to 0.000025 and training restarted.

After training to just over 20000 batches it became apparent that halving the learning rate did not help significantly. The models did not make visual progress, the GAN loss skyrocketed and the d2 loss went to zero.

Another issue is the size on disk - model checkpoints are being saved after every batch. The large number of checkpoints occupy almost 10 TB. To train further, additional disk space is needed. The plan is to stop training temporarily and generate frames for training videos for a number of interesting latent points up to 19000 batches. Then, a few earlier model checkpoints can be manually curated for the archive and the rest deleted. This will free up space to train for significantly longer.

#### Training frame sequences

Latent points for training sequences were chosen based on the following specimens:

1. 16500.28 - complete 1o 19000 frames
2. 18218.29 - complete 1o 19000 frames
3. 18218.3 - complete 1o 19000 frames
4. 16500.21 - running on GPU 2/screen 2
5. 18218.11 - running on GPU 0/screen 0
6. 18218.6 - running on GPU 1/screen 1
