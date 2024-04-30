using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class Phi2MLP : torch.nn.Module<Tensor, Tensor>
{
    private readonly Linear fc1;
    private readonly Linear fc2;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;

    public Phi2MLP(Phi2Config config)
        : base(nameof(Phi2MLP))
    {
        this.fc1 = torch.nn.Linear(config.HiddenSize, config.IntermediateSize, dtype: config.Dtype);
        this.fc2 = torch.nn.Linear(config.IntermediateSize, config.HiddenSize, dtype: config.Dtype);
        this.activation_fn = new NewGELUActivation();
    }
    public override Tensor forward(Tensor input)
    {
        using var input1 = this.fc1.forward(input);
        using var input2 = this.activation_fn.forward(input1);
        return this.fc2.forward(input2);
    }
}
  