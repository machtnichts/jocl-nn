package org.machtnichts.neural;

import org.jocl.*;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


import static org.jocl.CL.*;
import static org.jocl.CL.clCreateKernel;

public class Main {
  public final static int INPUT_LAYER_SIZE=8;
  public final static int OUTPUT_LAYER_SIZE=8;
  public final static int HIDDEN_LAYER_SIZE=8;

  public static void main(String[] args) throws IOException {
    int botSize = 512*512;

    double[] input = new double[INPUT_LAYER_SIZE];
    for(int i=0; i<input.length;i++)
      input[i] = Math.random();

    var bots = createNetworks(botSize);
    runJOCLTest(bots,input);
    runJavaTest(bots,input);
    runJavaTestSeq(bots,input);
  }

  private static void example() throws IOException {
    String programSource = Files.readString(Paths.get("src/main/resources/kernels/ArrayMul.cl"));

    int n = 10;
    double[] srcArrayA = new double[n];
    double[] srcArrayB = new double[n];
    double[] dstArray = new double[n];
    for (int i=0; i<n; i++)
    {
      srcArrayA[i] = i;
      srcArrayB[i] = i;
    }
    Pointer srcA = Pointer.to(srcArrayA);
    Pointer srcB = Pointer.to(srcArrayB);
    Pointer dst = Pointer.to(dstArray);

    final int platformIndex = 0;
    final long deviceType = CL_DEVICE_TYPE_ALL;
    final int deviceIndex = 0;

    // Enable exceptions and subsequently omit error checks in this sample
    CL.setExceptionsEnabled(true);

    // Obtain the number of platforms
    int[] numPlatformsArray = new int[1];
    clGetPlatformIDs(0, null, numPlatformsArray);
    int numPlatforms = numPlatformsArray[0];

    // Obtain a platform ID
    cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(platforms.length, platforms, null);
    cl_platform_id platform = platforms[platformIndex];

    // Initialize the context properties
    cl_context_properties contextProperties = new cl_context_properties();
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

    // Obtain the number of devices for the platform
    int[] numDevicesArray = new int[1];
    clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
    int numDevices = numDevicesArray[0];

    // Obtain a device ID
    cl_device_id[] devices = new cl_device_id[numDevices];
    clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
    cl_device_id device = devices[deviceIndex];

    // Create a context for the selected device
    cl_context context = clCreateContext(
        contextProperties, 1, new cl_device_id[]{device},
        null, null, null);

    // Create a command-queue for the selected device
    cl_queue_properties properties = new cl_queue_properties();
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
        context, device, properties, null);

    // Allocate the memory objects for the input- and output data
    cl_mem srcMemA = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        Sizeof.cl_double * n, srcA, null);
    cl_mem srcMemB = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        Sizeof.cl_double * n, srcB, null);
    cl_mem dstMem = clCreateBuffer(context,
        CL_MEM_READ_WRITE,
        Sizeof.cl_double * n, null, null);

    // Create the program from the source code
    cl_program program = clCreateProgramWithSource(context,
        1, new String[]{ programSource }, null, null);

    // Build the program
    clBuildProgram(program, 0, null, null, null, null);

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

    // Set the arguments for the kernel
    int a = 0;
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemA));
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemB));
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(dstMem));

    // Set the work-item dimensions
    long[] global_work_size = new long[]{n};

    // Execute the kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
        global_work_size, null, 0, null, null);

    // Read the output data
    clEnqueueReadBuffer(commandQueue, dstMem, CL_TRUE, 0,
        n * Sizeof.cl_double, dst, 0, null, null);

    // Release kernel, program, and memory objects
    clReleaseMemObject(srcMemA);
    clReleaseMemObject(srcMemB);
    clReleaseMemObject(dstMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    // Verify the result
    boolean passed = true;
    final float epsilon = 1e-7f;
    for (int i=0; i<n; i++)
    {
      double x = dstArray[i];
      double y = srcArrayA[i] * srcArrayB[i];
      boolean epsilonEqual = Math.abs(x - y) <= epsilon * Math.abs(x);
      if (!epsilonEqual)
      {
        passed = false;
        break;
      }
    }
    System.out.println("Test "+(passed?"PASSED":"FAILED"));
    System.out.println("Result: "+Arrays.toString(dstArray));
  }


  private static void runJOCLTest(final List<NeuralNetworkSimple> bots, final double[] input) throws IOException {
    Instant start = Instant.now();
    final int platformIndex = 0;
    final long deviceType = CL_DEVICE_TYPE_ALL;
    final int deviceIndex = 0;

    String programSource = Files.readString(Paths.get("src/main/resources/kernels/NN.cl"));
    // Enable exceptions and subsequently omit error checks in this sample
    CL.setExceptionsEnabled(true);

    // Obtain the number of platforms
    int[] numPlatformsArray = new int[1];
    clGetPlatformIDs(0, null, numPlatformsArray);
    int numPlatforms = numPlatformsArray[0];

    // Obtain a platform ID
    cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(platforms.length, platforms, null);
    cl_platform_id platform = platforms[platformIndex];

    // Initialize the context properties
    cl_context_properties contextProperties = new cl_context_properties();
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

    // Obtain the number of devices for the platform
    int[] numDevicesArray = new int[1];
    clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
    int numDevices = numDevicesArray[0];

    // Obtain a device ID
    cl_device_id[] devices = new cl_device_id[numDevices];
    clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
    cl_device_id device = devices[deviceIndex];

    // Create a context for the selected device
    cl_context context = clCreateContext(
        contextProperties, 1, new cl_device_id[]{device},
        null, null, null);

    // Create a command-queue for the selected device
    cl_queue_properties properties = new cl_queue_properties();
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
        context, device, properties, null);

    // Create the program from the source code
    cl_program program = clCreateProgramWithSource(context,
        1, new String[]{ programSource }, null, null);

    // Build the program
    clBuildProgram(program, 0, null, null, null, null);

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "calcNN", null);

    long setupDuration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("jocl setup duration: "+setupDuration + " ms");

    start = Instant.now();

    int populationSize = bots.size();
    int weightSize = INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE;
    double[] weights = new double[populationSize*weightSize];
    double[] outputArray = new double[populationSize*OUTPUT_LAYER_SIZE];

    int botIndex = 0;
    for(NeuralNetworkSimple nn:bots)
    {
      double[] botWeights = nn.copyOfWeights();
      System.arraycopy(botWeights,0,weights,botIndex*botWeights.length,botWeights.length);
      botIndex++;
    }

    Pointer inputPointer = Pointer.to(input);
    Pointer weightsPointer = Pointer.to(weights);
    Pointer outputPointer = Pointer.to(outputArray);

    // Allocate the memory objects for the input- and output data
    cl_mem srcMemInputArray = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (long) Sizeof.cl_double * input.length, inputPointer, null);
    cl_mem srcMemWeightsArray = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (long) Sizeof.cl_double * weights.length, weightsPointer, null);
    cl_mem dstMemOutputArray = clCreateBuffer(context,
        CL_MEM_READ_WRITE,
        (long) Sizeof.cl_double * outputArray.length, null, null);


    // Set the arguments for the kernel
    int a = 0;
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemInputArray));
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemWeightsArray));
    clSetKernelArg(kernel, a++, Sizeof.cl_int, Pointer.to(new int[]{ INPUT_LAYER_SIZE }));
    clSetKernelArg(kernel, a++, Sizeof.cl_int, Pointer.to(new int[]{ weightSize }));
    clSetKernelArg(kernel, a++, Sizeof.cl_int, Pointer.to(new int[]{ OUTPUT_LAYER_SIZE }));
    clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(dstMemOutputArray));

    // Set the work-item dimensions
    long[] global_work_size = new long[]{bots.size()};


    long uploadDuration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("jocl UPLOAD duration: "+uploadDuration + " ms");

    start = Instant.now();
    // Execute the kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
        global_work_size, null, 0, null, null);

    // Read the output data
    clEnqueueReadBuffer(commandQueue, dstMemOutputArray, CL_TRUE, 0,
        (long) outputArray.length * Sizeof.cl_double, outputPointer, 0, null, null);

    long duration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("java jocl duration: "+duration + " ms");

    start = Instant.now();
    // Release kernel, program, and memory objects
    clReleaseMemObject(srcMemInputArray);
    clReleaseMemObject(srcMemWeightsArray);
    clReleaseMemObject(dstMemOutputArray);
    long releaseDuration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("jocl RELEASE duration: "+releaseDuration + " ms");


    double[] lastResult = Arrays.copyOfRange(outputArray,outputArray.length-OUTPUT_LAYER_SIZE,outputArray.length);
    System.out.println("Last result: "+Arrays.toString(lastResult));

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
  }


  private static void runJavaTest(final List<NeuralNetworkSimple> bots, final double[] input)
  {
    Instant start = Instant.now();
    bots.parallelStream().forEach(bot -> bot.calculate(input));
    long duration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("java parallel duration: "+duration + " ms");
    System.out.println("Last result: "+Arrays.toString(bots.getLast().calculate(input)));
  }

  private static void runJavaTestSeq(final List<NeuralNetworkSimple> bots, final double[] input)
  {
    Instant start = Instant.now();
    bots.forEach(bot -> bot.calculate(input));
    long duration = Duration.between(start,Instant.now()).toMillis();
    System.out.println("java seq duration: "+duration + " ms");
    System.out.println("Last result: "+Arrays.toString(bots.getLast().calculate(input)));
  }



  private static List<NeuralNetworkSimple> createNetworks(int networkCount)
  {
    List<NeuralNetworkSimple> l = new ArrayList<>(networkCount);
    int inputLayerSize = INPUT_LAYER_SIZE;
    int outputLayerSize = OUTPUT_LAYER_SIZE;
    int hiddenLayerSize = HIDDEN_LAYER_SIZE;
    double[] weights = new double[inputLayerSize * hiddenLayerSize + hiddenLayerSize * outputLayerSize];

    for(int i=0; i<networkCount;i++)
    {
      for(int j=0; j<weights.length; j++){
        weights[j]=Math.random();
      }
      l.add(new NeuralNetworkSimple(inputLayerSize,hiddenLayerSize,outputLayerSize,Arrays.copyOf(weights,weights.length)));
    }
    return l;
  }




}
