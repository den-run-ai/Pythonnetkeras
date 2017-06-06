using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.Models
{
    public class KerasExecutionConfiguration<T> where T: MnistDatasetMlp
    {
        public string ModelDefinition { get; set; }

        public int Epochs { get; set; }

        public int BatchSize { get; set; }

        public string LossFunction { get; set; }

        public string OptimizationFunction { get; set; }
        
        public PyList Metrics { get; set; }

        public T MnistDataset { get; set; }
    }
}
