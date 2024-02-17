# Notes

## 1. Thoughts

### 1.1

This type of project is hard to experiment with - each new tweak requires a large amount of both data and time:

```text
$ du -sh data/*

14G     ./gan_output
400M    ./image_datasets
1.8G    ./specimens
5.0T    ./training_checkpoints
```

That's from one run, with one set of configuration variables and was generated over the course of ~4 days.

### 1.2

There are and uncountably infinite number of possible cities which can be generated from each model checkpoint.

### 1.3

Each model is a more or less 'good' method to find city-ness in an n 100 vector of random numbers.

### 1.4

The generated cities have a concept of nearness or similarity that is not physical or aesthetic, e.g. city [1.1, 1.2,...] is closer to city [1.2, 1.2,...] than it is to city [1.8, 1.2,...].

### 1.5

The model translates 100 random numbers into 3.2 million numbers which resemble a city when formatted as a jpeg image.

## 2. Training scratch

### 2.1. 2024-02-11 run

Learning rate was initially set at 0.000025. At 7000 batches the learning rate was manually updated to 0.00001. The learning rate was updated again at around 7800 batches to 0.0001 because the models had failed to progress visually. Prior archived runs used 0.0001 with good results.

Ran the model out to ~16k batches with hardly any progress. Seems like switching to a fast learning rate after the model had already started to converge was not helpful. In the future, try starting with a large learning rate and then decreasing it as training progresses.

Generated the training video out to about 9700 frames for archival purposes, but didn't keep any other artifacts.

### 2.2. 2024-02-17 run

Learning from the results of the 2024-02-11 training run, a learning rate schedule was implemented in order to start training with large learning rate and decrease it over time.
