__kernel void
calcNN(__global const double *input,
             __global const double *weights,
              const int inputSize,
              const int weightsSize,
              const int outputSize,
             __global double *output)
{
    int gid = get_global_id(0);
    const int hiddenSize = 8;
    double hiddenOutput[8];

    int weightsShift = gid*weightsSize;

    int weightIndex = weightsShift;
    for (int j = 0; j < hiddenSize; j++) {
      for (int i = 0; i < inputSize; i++) {
        hiddenOutput[j] += input[i] * weights[weightIndex];
        weightIndex++;
      }
    }

    for (int i = 0; i < hiddenSize; i++) {
      hiddenOutput[i] = 1.0 / (1 + exp(-hiddenOutput[i])) * 2.0 - 1.0;
      if (hiddenOutput[i] < 0) {
        hiddenOutput[i] = hiddenOutput[i] / 2.0;
      }
    }


    int outputShift = gid*outputSize;

    for (int j = 0; j < outputSize; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        output[outputShift+j] += hiddenOutput[i] * weights[weightIndex];
        weightIndex++;
      }
    }

    for (int i = 0; i < outputSize; i++) {
      output[outputShift+i] = 1.0 / (1.0 + exp(-output[outputShift+i]));
    }

//   for (int i = 0; i < outputSize; i++) {
//     output[i] = weights[i];
//   }

}