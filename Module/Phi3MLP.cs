using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using Phi.Module;

public class Phi3MLP : torch.nn.Module<Tensor, Tensor>
{
    private readonly PhiLinear gate_up_proj;
    private readonly PhiLinear down_proj;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;

    public Phi3MLP(Phi3Config config)
        : this(config.HiddenSize, config.IntermediateSize, config.HiddenAct, config.DType)
    {
    }

    public Phi3MLP(int hiddenSize, int intermediateSize, string hiddenAct, ScalarType dtype)
        : base(nameof(Phi3MLP))
    {
        this.gate_up_proj = new PhiLinear(hiddenSize, 2 * intermediateSize, hasBias: false, dtype: dtype);
        this.down_proj = new PhiLinear(intermediateSize, hiddenSize, hasBias: false, dtype: dtype);
        this.RegisterComponents();
        this.activation_fn = Utils.GetActivation(hiddenAct);
    }

    public override Tensor forward(Tensor input)
    {
        using var input1 = this.gate_up_proj.forward(input);
        var chunks = input1.chunk(2, dim: -1);
        var gate = chunks[0];
        var up_status = chunks[1];
        up_status = up_status * this.activation_fn.forward(gate);
        return this.down_proj.forward(up_status);
    }
}
