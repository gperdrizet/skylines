# Notes

## 1

This type of project is hard to experiment with - each new tweak requires a large amount of both data and time:

```text
$ du -sh data/*

14G     ./gan_output
400M    ./image_datasets
1.8G    ./specimens
5.0T    ./training_checkpoints
```

That's from one run, with one set of configuration variables and was generated over the course of ~4 days.

## 2

There are and uncountably infinite number of possible cities which can be generated from each model checkpoint.

## 3

Each model is a more or less 'good' method to find city-ness in an n 100 vector of random numbers.

## 4

The generated cities have a concept of nearness or similarity that is not physical or aesthetic, e.g. city [1.1, 1.2,...] is closer to city [1.2, 1.2,...] than it is to city [1.8, 1.2,...].

## 5

The model translates 100 random numbers into 3.2 million numbers which resemble a city when formatted as a jpeg image.
