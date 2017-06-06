using Mnist.Constants;
using Mnist.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Python.Runtime.Py;

namespace Mnist
{
    public class MnistCnnExecution : IExecute
    {
        private GILState python;
        private dynamic model;
        private KerasExecutionConfiguration<MnistDatasetCnn> configuration;

        public MnistCnnExecution(KerasExecutionConfiguration<MnistDatasetCnn> configuration)
        {
            this.configuration = configuration;
            ConfigureModules();
            model.compile(Py.kw(KerasConstants.Compile.Loss, configuration.LossFunction, KerasConstants.Compile.Optimizer, configuration.OptimizationFunction, KerasConstants.Compile.Metrics, configuration.Metrics));
        }

        public void Evaluate()
        {
            dynamic score = model.evaluate(configuration.MnistDataset.TestX, configuration.MnistDataset.TestY, Py.kw(KerasConstants.Evaluate.Verbose, 0));
        }

        public void Fit()
        {
            var validationData = new PyTuple(new PyObject[] { configuration.MnistDataset.TestX, configuration.MnistDataset.TestY });
            model.fit(configuration.MnistDataset.TrainX, configuration.MnistDataset.TrainY, Py.kw(KerasConstants.Fit.BatchSize, configuration.BatchSize, KerasConstants.Fit.Epochs, configuration.Epochs, KerasConstants.Fit.ValidationData, 1, KerasConstants.Fit.ValidationData, validationData));
        }

        private void ConfigureModules()
        {
            var kerasModelsModule = PythonEngine.ModuleFromString("keras.modules", "from keras.models import model_from_json");
            dynamic loadModule = kerasModelsModule.GetAttr("model_from_json");
            this.model = loadModule(configuration.ModelDefinition);
        }

    }
}
