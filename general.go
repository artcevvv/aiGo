package main

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

// Creating networks with the random-setted weights

func createNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return
}

// Predict- function for predicting values using the trained neural netw

func (net *Network) Predict(inputData []float64) mat.Matrix {
	// Forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
}

//Training n.n.

func (net *Network) Train(inputData []float64, targetData []float64) {
	// Forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// Find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrs := subtract(targets, finalOutputs)
	hiddenErrs := dot(net.outputWeights.T(), outputErrs)

	// Back propagation
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrs, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = add(net.hiddenWeights,
		scale(net.learningRate,
			dot(multiply(hiddenErrs, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

// Every function below if used for simplifying the proccess of working with the matrices

func sigmoid(r, c int, z float64) float64 {
	sig := 1.0 / (1 + math.Exp(-1*z))
	return sig
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m))
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// func addScalar(i float64, m mat.Matrix) mat.Matrix {
// 	r, c := m.Dims()
// 	b := r * c
// 	a := make([]float64, b)
// 	for x := 0; x < b; x++ {
// 		a[x] = i
// 	}
// 	n := mat.NewDense(r, c, a)
// 	return add(m, n)
// }

// Creating a random array(of the float64 data!!!) to fill the new dense(заряженная или плотная) matrix

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func save(net Network) {
	h, err := os.Create("data/hiddenweights.model")

	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	defer h.Close()

	o, err := os.Create("data/outputweights.model")
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
	defer o.Close()
}

func load(net *Network) {
	h, err := os.Open("data/hiddenweights.model")
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	defer h.Close()

	o, err := os.Open("data/outputweights.model")
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	defer o.Close()
}

// pretty matrix return

func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
