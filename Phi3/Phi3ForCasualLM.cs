using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace Phi;

public class Phi3ForCasualLM : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
    private readonly Phi3Config config;
    private readonly Phi3Model model;
    private readonly Linear lm_head;

    public Phi3ForCasualLM(Phi3Config config)
        : base(nameof(Phi3ForCasualLM))
    {
        this.config = config;
        this.model = new Phi3Model(config);
        this.lm_head = torch.nn.Linear(config.HiddenSize, config.VocabSize, dtype: config.DType, hasBias: false);

        this.RegisterComponents();
    }

    public override CasualLMModelOutput forward(CasualLMModelInput input)
    {
        var outputs = this.model.forward(input);
        var logits = this.lm_head.forward(outputs.last_hidden_state);
        logits = logits.to_type(ScalarType.Float32);
        outputs.logits = logits;

        return outputs;
    }

    public static Phi3ForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "model.safetensors.index.json",
        ScalarType torchDtype = ScalarType.BFloat16,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = torchDtype;
        var phi = new Phi3ForCasualLM(modelConfig);
        var loadedParameters = new Dictionary<string, bool>();
        phi.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: false, loadedParameters: loadedParameters, useTqdm: false);
        phi = phi.to(device);
        phi.eval();

        return phi;
    }
}
