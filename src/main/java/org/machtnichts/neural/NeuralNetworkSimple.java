package org.machtnichts.neural;

import java.util.Arrays;

public class NeuralNetworkSimple implements Cloneable {
  private final int inputLayerSize;
  private final int outputLayerSize;
  private final int hiddenLayerSize;
  private final double[] weights;

  public NeuralNetworkSimple(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double[] weights) {
    if (weights.length != inputLayerSize * hiddenLayerSize + hiddenLayerSize * outputLayerSize)
      throw new IllegalArgumentException("wrong weights dimension");
    this.inputLayerSize = inputLayerSize;
    this.outputLayerSize = outputLayerSize;
    this.hiddenLayerSize = hiddenLayerSize;
    this.weights = weights;
  }

  public double[] calculate2(double[] input) {
    //upload weights and input
    //run calc
    //download output
    double[] output = new double[outputLayerSize];
    return output;
  }

  public double[] calculate(double[] input) {
    double[] hiddenOutput = new double[hiddenLayerSize];
    int weightIndex = 0;
    for (int j = 0; j < hiddenLayerSize; j++) {
      for (int i = 0; i < inputLayerSize; i++) {
        hiddenOutput[j] += input[i] * weights[weightIndex];
        weightIndex++;
      }
    }

    for (int i = 0; i < hiddenOutput.length; i++) {
      hiddenOutput[i] = hiddenSigmoid(hiddenOutput[i]);
      if (hiddenOutput[i] < 0) {
        hiddenOutput[i] = hiddenOutput[i] / 2F;
      }
    }

    double[] output = new double[outputLayerSize];
    for (int j = 0; j < outputLayerSize; j++) {
      for (int i = 0; i < hiddenLayerSize; i++) {
        output[j] += hiddenOutput[i] * weights[weightIndex];
        weightIndex++;
      }
    }
    for (int j = 0; j < outputLayerSize; j++) {
      output[j] = outputSigmoid(output[j]);
    }
    return output;
  }

  public double[] copyOfWeights()
  {
    return Arrays.copyOf(weights,weights.length);
  }

  public NeuralNetworkSimple clone() {
    return new NeuralNetworkSimple(inputLayerSize, hiddenLayerSize, outputLayerSize, Arrays.copyOf(weights, weights.length));
  }

  public static double hiddenSigmoid(double x) {
    return (1D / (1 + Math.exp(-x))) * 2 - 1;
  }

  public static double outputSigmoid(double x) {
    return (1D / (1 + Math.exp(-x)));
  }
}
