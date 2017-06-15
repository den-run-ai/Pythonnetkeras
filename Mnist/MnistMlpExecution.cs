using Mnist.Constants;
using Mnist.Models;
using Python.Runtime;
using System;
using static Python.Runtime.Py;

namespace Mnist
{
    public interface IExecute
    {
        void Fit();
        void Evaluate();
    }


    public class MnistMlpExecution : IDisposable, IExecute
    {
        private dynamic model;
        private GILState python;
        private KerasExecutionConfiguration<MnistDatasetMlp> configuration;

        public MnistMlpExecution(KerasExecutionConfiguration<MnistDatasetMlp> configuration)
        {
            this.configuration = configuration;
            ConfigureModules();
            this.model.compile(Py.kw(KerasConstants.Compile.Loss, configuration.LossFunction, KerasConstants.Compile.Optimizer, configuration.OptimizationFunction, KerasConstants.Compile.Metrics, configuration.Metrics));
        }

        public void Fit()
        {
            var validationData = new PyTuple(new PyObject[] { configuration.MnistDataset.TestX, configuration.MnistDataset.TestY });
            this.model.fit(configuration.MnistDataset.TrainX, configuration.MnistDataset.TrainY, Py.kw(KerasConstants.Fit.BatchSize, configuration.BatchSize, KerasConstants.Fit.Epochs, configuration.Epochs, KerasConstants.Fit.Verbose, 1, KerasConstants.Fit.ValidationData, validationData));
        }

        public void Evaluate()
        {
            dynamic score = this.model.evaluate(configuration.MnistDataset.TestX, configuration.MnistDataset.TestY, Py.kw(KerasConstants.Evaluate.Verbose, 1));
            Console.WriteLine(score[0]);
            Console.WriteLine(score[1]);
        }

        public void Dispose()
        {
            this.python.Dispose();
            GC.SuppressFinalize(this);
        }

        public void ConfigureModules()
        {
            var kerasModelsModule = PythonEngine.ModuleFromString("keras.modules", "from keras.models import model_from_json");
            dynamic loadModule = kerasModelsModule.GetAttr("model_from_json");
            this.model = loadModule(configuration.ModelDefinition);
        }
    }
}
