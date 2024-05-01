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
using FluentAssertions;

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
        var state_dict_str = model.Peek();
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
        var input = new CasualLMModelInput(inputIds, attentionMask, past_key_values_length: 0);
        var output = model.forward(input);
        var outputTokenIds = output.last_hidden_state;
        var outputLogits = output.logits;

        var outputTokenIdsStr = outputTokenIds.Peek("output");
        var outputLogitsStr = outputLogits.Peek("logits");

        var sb = new StringBuilder();
        sb.AppendLine(outputTokenIdsStr);
        sb.AppendLine(outputLogitsStr);

        Approvals.Verify(sb.ToString());
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task TokenizerTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
        var tokenizer = Phi2Tokenizer.FromPretrained(modelWeightFolder);
        tokenizer.EosId.Should().Be(50256);
        tokenizer.BosId.Should().Be(50256);
        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?"
        };
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var tokenized = tokenizer.Encode(message, true, false);
            var tokenized_str = string.Join(", ", tokenized.Select(x => x.ToString()));

            sb.AppendLine(tokenized_str);
        }
        Approvals.Verify(sb.ToString());
    }
}
