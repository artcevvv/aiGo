package main

import (
	"flag"
	"fmt"
)

func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 200 hidden neurons
	// 10 outputs
	// LR = 0.1
	net := createNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		// chillin
	}

	if *file != "" {
		printImage(getImage(*file))
		load(&net)
		fmt.Println("prediction:", predictFromImage(net, *file))
	}
}
