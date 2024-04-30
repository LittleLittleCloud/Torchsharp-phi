using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace Phi.Module;

public class Phi3RMSNorm : torch.nn.Module<Tensor, Tensor>
{
    private int _dim;
    private float _eps;
    private Parameter weight;

    public Phi3RMSNorm(
        int hiddenSize,
        float eps = 1e-6f)
        : base(nameof(Phi3RMSNorm))
    {
        this._dim = hiddenSize;
        this._eps = eps;

        // the gamma scalar
        this.weight = torch.nn.Parameter(torch.ones(this._dim, dtype: ScalarType.Float32));
    }

    private Tensor Norm(Tensor x)
    {
        // (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        // rsqrt = 1 / sqrt
        var output = x * torch.rsqrt(x.pow(2).mean([-1L], keepdim: true) + this._eps);
        return output;
    }

    public override Tensor forward(Tensor input)
    {
        // needs higher precision for the norm so convert to float32
        // (B, Seq_Len, Dim)
        var normed = this.Norm(input.to_type(ScalarType.Float32)).type_as(input);
        // (B, Seq_Len, Dim) * (Dim) = (B, Seq_Len, Dim)
        var output = this.weight * normed;

        return output;
    }
}
