using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using ApprovalTests;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using Xunit;
using TorchSharp;

namespace Phi.Tests;

public class Phi2Test
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task LoadSafeTensorShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
        var model = Phi2ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16, checkPointName: "model.safetensors.index.json");
        var state_dict_str = model.Model.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ForwardTest()
    {
        // create dummy input id with 128 length and attention mask
        var device = "cuda";
        var inputIds = torch.arange(128, dtype: ScalarType.Int64, device: device).unsqueeze(0);
        var attentionMask = torch.ones(1, 128, device: device);
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
        var model = Phi2ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16, checkPointName: "model.safetensors.index.json", device: "cuda");

        var (outputTokenIds, outputLogits) = model.Generate(inputIds, attentionMask);

        var outputTokenIdsStr = outputTokenIds.Peek("output");
        var outputLogitsStr = outputLogits.Peek("logits");

        var sb = new StringBuilder();
        sb.AppendLine(outputTokenIdsStr);
        sb.AppendLine(outputLogitsStr);

        Approvals.Verify(sb.ToString());
    }
}
