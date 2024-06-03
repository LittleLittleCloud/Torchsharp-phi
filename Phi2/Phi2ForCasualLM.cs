using Phi;
using System.CodeDom;
using System.Text.Json;
using System.Text.Json.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

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

public class Phi2ForCasualLM : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
    private readonly Phi2Model model;

    private readonly Linear lm_head;

    public Phi2ForCasualLM(Phi2Model model)
        : base(nameof(Phi2ForCasualLM))
    {
        this.model = model;
        this.lm_head = nn.Linear(model.Config.HiddenSize, model.Config.VocabSize, dtype: model.Config.Dtype);
        this.RegisterComponents();
    }

    public override CasualLMModelOutput forward(CasualLMModelInput input) // use_cache, output_attentions, output_hidden_states
    {
        var inputIds = input.input_ids;
        var attentionMask = input.attention_mask;
        var pastKeyValueLength = input.past_key_values_length;
        var positionIds = input.position_ids;
        var inputEmbeddings = input.inputs_embeds;
        var options = (input.output_attentions, input.output_hidden_states, false);
        var output = this.model.forward(inputIds, attentionMask, pastKeyValueLength, positionIds, inputEmbeddings, options);
        var hidden_state = output.Item1;

        var lm_logits = this.lm_head.forward(hidden_state);

        return new CasualLMModelOutput(last_hidden_state: hidden_state, legits: lm_logits);
    }

    public static Phi2ForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "phi-2.pt",
        ScalarType torchDtype = ScalarType.Float32,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi2Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.Dtype = torchDtype;
        var phi = new Phi2Model(modelConfig);
        var wrapper = new Phi2ForCasualLM(phi);
        var loadedParameters = new Dictionary<string, bool>();
        wrapper.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: true, loadedParameters: loadedParameters);
        wrapper = wrapper.to(device);
        wrapper.eval();
        
        return wrapper;
    }
}
  