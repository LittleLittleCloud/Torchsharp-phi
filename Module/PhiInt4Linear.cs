using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi.Module;

public class PhiInt4Linear : PhiLinear, IQuantizeModule
{
    public PhiInt4Linear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(inFeatures, outFeatures, hasBias, dtype, device)
    {
    }

    public void Quantize()
    {
        using var _ = NewDisposeScope();
        var timer = new System.Diagnostics.Stopwatch();
        Console.WriteLine("Quantize start");
        timer.Start();
        // scale and zero point on vector-wise
        // scale = 15 / max(weight, axis=1) - min(weight, axis=1)
        var scale = 15 / (torch.max(this.weight, 1).values - torch.min(this.weight, 1).values);

        // zero point = - scale * min(weight, axis=1) - 8
        var zeroPoint = - scale * torch.min(this.weight, 1).values - 8;
        // round zero point to nearest integer
        zeroPoint = torch.round(zeroPoint);
        var _4bitWeight = torch.round(this.weight * scale.view(-1, 1) + zeroPoint.view(-1, 1)).to(torch.int8);

        zeroPoint = (zeroPoint + 8).to(torch.uint8);
        _4bitWeight = (_4bitWeight + 8).view(-1).to(torch.uint8);

        // torch doesn't provide int4, so we use int8 as placeholder
        // and foreach int8, we save two int4, e.g. 0b1010 -> 0b10, 0b10
        var placeHolderDim = this.outFeatures / 2 + this.outFeatures % 2;
        var zpPlaceHolder = zeroPoint[..placeHolderDim];
        zpPlaceHolder = zpPlaceHolder * 16 + zeroPoint[placeHolderDim..];

        // assert zero point is in range [-128, 127]
        //if (torch.any(this.zeroPoint < -128).item<bool>() || torch.any(this.zeroPoint > 127).item<bool>())
        //{
        //    throw new Exception("Zero point is out of range [-128, 127]");
        //}

        // quantize weight
        var _4bitWeightPlaceHolderDim =Convert.ToInt32(_4bitWeight.size(0) / 2 + _4bitWeight.size(0) % 2);
        var _4bitWeightPlaceHolder = _4bitWeight[.._4bitWeightPlaceHolderDim];
        _4bitWeightPlaceHolder = _4bitWeightPlaceHolder * 16 + _4bitWeight[_4bitWeightPlaceHolderDim..];

        // assert weight is in range [-128, 127]
        //if (torch.any(this._8bitWeight < -128).item<bool>() || torch.any(this._8bitWeight > 127).item<bool>())
        //{
        //    throw new Exception("Weight is out of range [-128, 127]");
        //}

        // dispose float32 weight
        this.weight.Dispose();
        this.weight = null;

        this._internal_buffers.Remove("weight");
        this.register_buffer("4bit_weight", _4bitWeightPlaceHolder.MoveToOuterDisposeScope());
        this.register_buffer("zeroPoint", zpPlaceHolder.MoveToOuterDisposeScope());
        this.register_buffer("scale", scale.MoveToOuterDisposeScope());
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
            var weight = this.get_buffer("4bit_weight");
            var weightLower = weight % 16;
            var weightUpper = weight / 16;
            weight = torch.cat([weightUpper, weightLower], 0).to(ScalarType.Float32);
            weight = weight.view(this.outFeatures, this.inFeatures);
            weight -= 8;
            var zeroPoint = this.get_buffer("zeroPoint");
            var zeroPointLower = zeroPoint % 16;
            var zeroPointUpper = zeroPoint / 16;
            zeroPoint = torch.cat([zeroPointUpper, zeroPointLower], 0).to(ScalarType.Float32);
            zeroPoint -= 8;
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
