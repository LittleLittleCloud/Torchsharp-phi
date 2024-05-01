using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

public class Phi3MLP : torch.nn.Module<Tensor, Tensor>
{
    private readonly Linear gate_up_proj;
    private readonly Linear down_proj;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;

    public Phi3MLP(Phi3Config config)
        : base(nameof(Phi3MLP))
    {
        this.gate_up_proj = torch.nn.Linear(config.HiddenSize, 2 * config.IntermediateSize, hasBias: false, dtype: config.DType);
        this.down_proj = torch.nn.Linear(config.IntermediateSize, config.HiddenSize, hasBias: false, dtype: config.DType);
        this.activation_fn = Utils.GetActivation(config.HiddenAct);
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
