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
    }

    public override Tensor forward(Tensor input)
    {
        using var dispose = torch.NewDisposeScope();
        if (this._internal_buffers.ContainsKey("weight"))
        {
            return base.forward(input);
        }
        else
        {
            var weight = this.get_buffer("8bit_weight").to(ScalarType.Float32).T;
            var zeroPoint = this.get_buffer("zeroPoint").to(ScalarType.Float32);
            var scale = this.get_buffer("scale").to(ScalarType.Float32);
            if (input.shape.Length == 3)
            {
                input = input.view(-1, input.shape[2]);
            }
            // quantize input
            // b * seq * channel
            var inputScale = 255 / (torch.max(input, -1).values - torch.min(input, -1).values);
            // b * seq
            var inputZeroPoint = -inputScale * torch.min(input, -1).values - 128;
            inputZeroPoint = torch.round(inputZeroPoint);

            // b * seq * channel
            var _8bitInput = torch.round(input * inputScale.view(-1, 1) + inputZeroPoint.view(-1, 1)).to(torch.float32);

            // matmul
            // input * weight = (_8bitInput - inputZeroPoint) * (_8bitWeight - zeroPoint) / (inputScale * scale)
            //               = _8bitInput * _8bitWeight - _8bitInput * zeroPoint - inputZeroPoint * _8bitWeight + inputZeroPoint * zeroPoint / (inputScale * scale)
            // output shape: [b, seq, hidden]
            // seq * hidden because _8bitWeight is [channel, hidden]
            var result = torch.matmul(_8bitInput, weight);

            // _8bitInput: [b, seq, channel]
            // zeroPoint: [channel]
            var zeroPointExpanded = zeroPoint!.unsqueeze(0).expand(this.inFeatures, -1).to(ScalarType.Float32);
            result -= torch.matmul(_8bitInput, zeroPointExpanded);

            // inputZeroPoint: [b, seq]
            // _8bitWeight: [channel, hidden]
            // inputZeroPointExpanded: [b, seq, hidden]
            var inputZeroPointExpanded = inputZeroPoint.unsqueeze(1).expand(-1, inFeatures).to(torch.float32);
            result += torch.matmul(inputZeroPointExpanded, zeroPointExpanded);

            result /= (inputScale.view(-1, 1) * scale!.view(1, -1));
            //// use float32
            //var input2 = input.to_type(ScalarType.Float32);
            //var weight2 = this.weight.to_type(ScalarType.Float32);
            //var result = torch.matmul(input2, weight2.t());

            if (this.bias is not null)
            {
                result = result + this.bias.to_type(ScalarType.Float32);
            }
            result.Peek("result");
            return result.to_type(input.dtype).MoveToOuterDisposeScope();
        }
    }
}
