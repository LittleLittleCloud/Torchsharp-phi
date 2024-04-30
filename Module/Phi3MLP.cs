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
        this.activation_fn = new NewGELUActivation();
    }
    public override Tensor forward(Tensor input)
    {
        using var input1 = this.gate_up_proj.forward(input);
        using var input2 = this.activation_fn.forward(input1);
        return this.down_proj.forward(input2);
    }
}
