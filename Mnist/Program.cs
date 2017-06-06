using Mnist.Constants;
using Mnist.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            TrainUsingMlpNetwork();
            TrainUsingCnnNetwork();
            
        }

        public static string RedirectOutput()
        {
            return
                   "import sys\n" +
                   "from io import StringIO\n" +
                   "sys.stdout = mystdout = StringIO()\n" +
                   "sys.stdout.flush()\n" +
                   "sys.stderr = mystderr = StringIO()\n" +
                   "sys.stderr.flush()\n";
        }

        public static void TrainUsingMlpNetwork()
        {
            using (var py = Py.GIL())
            {
                PythonEngine.Exec(RedirectOutput());
                int numberOfClasses = 10;
                int vectorLength = 784;
                int trainVectorCount = 60000;
                int testVectorCount = 10000;
                int epoch = 20;
                int batchSize = 128;
                var data = new MnistDataSetGenerator(numberOfClasses, vectorLength, trainVectorCount, testVectorCount);
                var mnistDatasetForMlp = data.GetMnistDataSet();

                PyTuple inputShape = new PyTuple(new PyObject[] { new PyInt(vectorLength) });
                var modelDefinition = new MnistMlpNetwokConfiguration().AddDense(512, KerasConstants.ActivationFunction.Relu, inputShape).AddDropout(0.2).AddDense(512, KerasConstants.ActivationFunction.Relu).AddDropout(0.2).AddDense(10, KerasConstants.ActivationFunction.Softmax).Build();

                var executionConfiguration = new KerasExecutionConfiguration<MnistDatasetMlp>();
                executionConfiguration.ModelDefinition = modelDefinition;
                executionConfiguration.MnistDataset = mnistDatasetForMlp;
                executionConfiguration.OptimizationFunction = KerasConstants.ActivationFunction.RmsProp;
                executionConfiguration.LossFunction = KerasConstants.LossFunction.CategoricalCrossentropy;
                executionConfiguration.Metrics = new PyList(new PyObject[] { new PyString(KerasConstants.Metrics.Accuracy) });
                executionConfiguration.BatchSize = batchSize;
                executionConfiguration.Epochs = epoch;
                var mnistMlpExec = new MnistMlpExecution(executionConfiguration);
                mnistMlpExec.Fit();
                mnistMlpExec.Evaluate();
            }
        }

        public static void TrainUsingCnnNetwork()
        {
            using (var py = Py.GIL())
            {

                int imageX = 28;
                int imageY = 28;
                int batchSize = 128;
                int numberOfClasses = 10;
                int epochs = 2;

                PythonEngine.Exec(RedirectOutput());
                var mnistCnnDatasetGenerator = new MnistDataSetGenerator(numberOfClasses, imageX, imageY);
                var mnistDatasetForCnn = mnistCnnDatasetGenerator.GetMnistDataSetForCnnNetwork();

                var serializedModel = new MnistCnnNetworkConfiguration()
                    .AddConvolutional2DLayer(64, new PyTuple(new PyObject[] { new PyInt(3), new PyInt(3) }), KerasConstants.ActivationFunction.Relu, mnistDatasetForCnn.InputShape)
                    .AddConvolutional2DLayer(64, new PyTuple(new PyObject[] { new PyInt(3), new PyInt(3) }), KerasConstants.ActivationFunction.Relu)
                    .AddMaxPooling2DLayer(new PyTuple(new PyObject[] { new PyInt(2), new PyInt(2) }))
                    .AddDropoutLayer(0.25)
                    .AddFlattenLayer()
                    .AddDenseLayer(128, KerasConstants.ActivationFunction.Relu)
                    .AddDropoutLayer(0.5)
                    .AddDenseLayer(numberOfClasses, KerasConstants.ActivationFunction.Softmax).Build();


                var executionConfiguration = new KerasExecutionConfiguration<MnistDatasetCnn>();
                executionConfiguration.BatchSize = batchSize;
                executionConfiguration.Epochs = epochs;
                executionConfiguration.ModelDefinition = serializedModel;
                executionConfiguration.LossFunction = KerasConstants.LossFunction.CategoricalCrossentropy;
                executionConfiguration.OptimizationFunction = KerasConstants.ActivationFunction.Adadelta;
                executionConfiguration.Metrics = new PyList(new PyObject[] { new PyString(KerasConstants.Metrics.Accuracy) });
                executionConfiguration.MnistDataset = mnistDatasetForCnn;

                var mnistExecution = new MnistCnnExecution(executionConfiguration);
                mnistExecution.Fit();
                mnistExecution.Evaluate();
            }
        }
    }
}
