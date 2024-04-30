using System.Text.Json.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;


public class PhiLinear : nn.Module<Tensor, Tensor>
{
    private readonly Tensor weight;
    private readonly Tensor? bias;
    private int inFeatures;
    private int outFeatures;

    public PhiLinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32)
        : base(nameof(PhiLinear))
    {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = torch.randn(outFeatures, inFeatures, dtype: dtype);

        if (hasBias)
        {
            this.bias = torch.randn(outFeatures, dtype: dtype);
            this.RegisterComponents();
        }
        else
        {
            this.RegisterComponents();
        }
    }

    public override Tensor forward(Tensor input)
    {
        using var dispose = torch.NewDisposeScope();

        // use float32
        var input2 = input.to_type(ScalarType.Float32);
        var weight2 = this.weight.to_type(ScalarType.Float32);
        var result = torch.matmul(input2, weight2.t());

        if (this.bias is not null)
        {
            result = result + this.bias.to_type(ScalarType.Float32);
        }

        return result.to_type(input.dtype).MoveToOuterDisposeScope();
    }
}

public class NewGELUActivation : torch.nn.Module<Tensor, Tensor>
{
    public NewGELUActivation()
        : base(nameof(NewGELUActivation))
    {
    }

    public override Tensor forward(Tensor input)
    {
        using var result = 0.044715 * torch.pow(input, 3.0);
        using var result2 = result + input;
        using var result3 = Math.Sqrt(2.0 / Math.PI) * result2;
        using var result4 = torch.tanh(result3);
        using var result5 = 1.0 + result4;
        return 0.5 * input * result5;
    }
}

public class PhiModelInferenceWrapper : nn.Module<
    Tensor, // input_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    Tensor?, // position_ids
    Tensor?, //input embeddings
    (
        bool, // use_cache
        bool, // output_attentions
        bool // output_hidden_states
    ),
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly Phi2Model model;

    private readonly Linear lm_head;

    public PhiModelInferenceWrapper(Phi2Model model)
        : base(nameof(PhiModelInferenceWrapper))
    {
        this.model = model;
        this.lm_head = nn.Linear(model.Config.HiddenSize, model.Config.VocabSize, dtype: model.Config.Dtype);
        this.RegisterComponents();
    }

    public override (Tensor, Tensor?, Tensor?) forward(
        Tensor inputIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        Tensor? positionIds = null,
        Tensor? inputEmbeddings = null,
        (bool, bool, bool) options = default) // use_cache, output_attentions, output_hidden_states
    {
        var output = this.model.forward(inputIds, attentionMask, pastKeyValueLength, positionIds, inputEmbeddings, options);
        var hidden_state = output.Item1;

        var lm_logits = this.lm_head.forward(hidden_state);

        return (lm_logits, output.Item2, output.Item3);
    }
}
  