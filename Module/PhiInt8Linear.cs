using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi.Module;

public class PhiInt8Linear : PhiLinear, IQuantizeModule
{
    //private Tensor? scale;
    //private Tensor? zeroPoint;
    //private Tensor? _8bitWeight;

    public PhiInt8Linear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(inFeatures, outFeatures, hasBias, dtype, device)
    {
    }

    public void Quantize()
    {
        var timer = new System.Diagnostics.Stopwatch();
        Console.WriteLine("Quantize start");
        timer.Start();
        // scale and zero point on vector-wise
        // scale = 255 / max(weight, axis=1) - min(weight, axis=1)
        var scale = 255 / (torch.max(this.weight, 1).values - torch.min(this.weight, 1).values);

        // zero point = - scale * min(weight, axis=1) - 128
        var zeroPoint = - scale * torch.min(this.weight, 1).values - 128;
        // round zero point to nearest integer
        zeroPoint = torch.round(zeroPoint).to(torch.int8);

        // assert zero point is in range [-128, 127]
        //if (torch.any(this.zeroPoint < -128).item<bool>() || torch.any(this.zeroPoint > 127).item<bool>())
        //{
        //    throw new Exception("Zero point is out of range [-128, 127]");
        //}

        // quantize weight
        var _8bitWeight = torch.round(this.weight * scale.view(-1, 1)+ zeroPoint.view(-1, 1)).to(torch.int8);

        // assert weight is in range [-128, 127]
        //if (torch.any(this._8bitWeight < -128).item<bool>() || torch.any(this._8bitWeight > 127).item<bool>())
        //{
        //    throw new Exception("Weight is out of range [-128, 127]");
        //}

        // dispose float32 weight
        this.weight.Dispose();
        this.weight = null;

        this._internal_buffers.Remove("weight");
        this.register_buffer("8bit_weight", _8bitWeight);
        this.register_buffer("zeroPoint", zeroPoint);
        this.register_buffer("scale", scale);
        timer.Stop();
        Console.WriteLine($"Quantize end, elapsed time: {timer.ElapsedMilliseconds} ms");
    }

    public override Tensor forward(Tensor input)
    {
        if (this._internal_buffers.ContainsKey("weight"))
        {
            return base.forward(input);
        }
        else
        {
            using var dispose = torch.NewDisposeScope();
            var weight = this.get_buffer("8bit_weight").to(ScalarType.Float32);
            var zeroPoint = this.get_buffer("zeroPoint").to(ScalarType.Float32);
            var scale = this.get_buffer("scale").to(ScalarType.Float32);
            var restoreWeight = (weight - zeroPoint.view(-1, 1)) / scale.view(-1, 1);
            // use float32
            var result = torch.matmul(input.to(ScalarType.Float32), restoreWeight.T);

            if (this.bias is not null)
            {
                result = result + this.bias.to_type(ScalarType.Float32);
            }

            //result.Peek("result");
            return result.to_type(input.dtype).MoveToOuterDisposeScope();
        }
    }
}
