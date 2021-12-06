CONVOLUTIONAL NEURAL NETWORK TESTS
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

I started my tests by creating the simplest "working" convolution network consisting of one Conv2D(filters=5, kernel_size=3) and one MaxPooling2D(pool_size=2) layer, which gave accuracy about 0.05.
First try was increasing filters and hidden layer size, which didn't give any change.
I've got first noticable change after adding another 2 Conv2D layers and one Pooling - accuracy increased to 0.68. Adding one more Convolution+Pooling layer gave me even better result.

After experimenting with the values and number of layers, I came to the following conclusions:
- increasing number of filters in subsequent Conv2D layers gives better performance and accuracy
- higher number of filters gives better accuracy, up to certain point, when the learning time starts to increase significantly, without improving accuracy
- kernel_size higher than 3 slows down learning without accuracy improvement.
- higher pooling shape size speeds up learning, but gives less accuracy - it is much better to have more pairs of Conv2d+Pooling layers than higher Pooling shape numbers
- adding Dense layers may improve accuracy, but slows down learning process. In this case adding just one additional layer gave small improvement, but adding more doesn't. Higher size gives better accuracy than higher number of hidden layers. Number and settings of Convolution+Pooling pairs are more significant for better results
- Dropout 0.2 - 0.5 gives similar results. 0.0 makes loss significantly higher. Dropout higher than 0.5 gives much lower accuracy

I also tried using GlobalMaxPooling2d instead of Flatten layer, which gave me slightly worse results.