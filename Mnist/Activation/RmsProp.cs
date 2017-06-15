using Mnist.Constants;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Python.Runtime.Py;

namespace Mnist.Activation
{
    public class OptimizationFunction
    {
        public string Name { get; set; }

        public dynamic OptimizeFunction { get; set; } 
    }

    public class RmsProp
    {
        private float learningRate;

        private float rho;

        private float epsilion;

        private float decay;

        public RmsProp(float learningRate = 0.001f, float rho = 0.9f, float epsilion = 0.0000000001f, float decay = 0.0f) : base()
        {
            this.learningRate = learningRate;
            this.rho = rho;
            this.epsilion = epsilion;
            this.decay = decay;
        }

        public OptimizationFunction Create()
        {
            using(var py = Py.GIL())
            {
                dynamic keras = Py.Import("keras");
                var result = new OptimizationFunction();
                result.Name = KerasConstants.ActivationFunction.RmsProp;
                result.OptimizeFunction = keras.optimizers.RMSprop(
                    Py.kw(KerasConstants.ActivationFunctionParameters.LearningRatio, this.learningRate,
                    KerasConstants.ActivationFunctionParameters.Rho, this.rho,
                    KerasConstants.ActivationFunctionParameters.Epsilon, this.epsilion,
                    KerasConstants.ActivationFunctionParameters.Decay, this.decay));
                return result;

            }
        }

    }

    public class AdaDelta
    {
        private float learningRate;

        private float rho;

        private float epsilion;

        private float decay;

        public AdaDelta(float learningRate = 1.0f, float rho = 0.95f, float epsilion = 0.0000000001f, float decay = 0.0f)
        {
            this.learningRate = learningRate;
            this.rho = rho;
            this.epsilion = epsilion;
            this.decay = decay;
        }

        public OptimizationFunction Create()
        {
            using (var py = Py.GIL())
            {
                dynamic keras = Py.Import("keras");
                var result = new OptimizationFunction();
                result.Name = KerasConstants.ActivationFunction.Adadelta;
                result.OptimizeFunction = keras.optimizers.Adadelta(
                    Py.kw(KerasConstants.ActivationFunctionParameters.LearningRatio, this.learningRate,
                    KerasConstants.ActivationFunctionParameters.Rho, this.rho,
                    KerasConstants.ActivationFunctionParameters.Epsilon, this.epsilion,
                    KerasConstants.ActivationFunctionParameters.Decay, this.decay));
                return result;

            }
        }

    }
}
