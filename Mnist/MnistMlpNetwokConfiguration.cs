using Mnist.Constants;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Python.Runtime.Py;

namespace Mnist
{
    public class MnistMlpNetwokConfiguration : IDisposable
    {
        private GILState python;
        private dynamic keras;
        private dynamic model;
        private dynamic denseLayer;
        private dynamic dropout;

        public MnistMlpNetwokConfiguration()
        {
            this.python = Py.GIL();
            ConfigureModules();
        }
        
        public MnistMlpNetwokConfiguration AddDropout(double ratio)
        {
            model.add(dropout(ratio));
            return this;
        }

        public MnistMlpNetwokConfiguration AddDense(int numberOfNeurons, string activationFunction, PyTuple inputShape = null)
        {
            dynamic denseLayerConfiguration = null;
            if(inputShape != null)
            {
                denseLayerConfiguration = this.denseLayer(numberOfNeurons, Py.kw(KerasConstants.Layer.Activation, activationFunction, KerasConstants.Layer.InputShape, inputShape));
            }
            else
            {
                denseLayerConfiguration = this.denseLayer(numberOfNeurons, Py.kw(KerasConstants.Layer.Activation, activationFunction));
            }

            model.add(denseLayerConfiguration);
            return this;
        }


        public string Build()
        {
            return model.to_json();
        }

        public void Dispose()
        {
            this.python.Dispose();
            GC.SuppressFinalize(this);
        }

        private void ConfigureModules()
        {
            PyObject sequentialModule = PythonEngine.ModuleFromString("sequential", "from keras.models import Sequential");
            PyObject layersModule = PythonEngine.ModuleFromString("layers", "from keras.layers import Dense, Dropout");
            dynamic Sequential = sequentialModule.GetAttr("Sequential");
            this.keras = Py.Import("keras");
            this.dropout = layersModule.GetAttr("Dropout");
            this.denseLayer = layersModule.GetAttr("Dense");
            this.model = Sequential();
        }
    }
}
