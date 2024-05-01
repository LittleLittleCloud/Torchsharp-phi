using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using ApprovalTests;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using Xunit;
using FluentAssertions;

namespace Phi.Tests;

public class Phi3Tests
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var model = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = model.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task TokenizerTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var tokenizer = LLama2Tokenizer.FromPretrained(modelWeightFolder);
        tokenizer.BosId.Should().Be(1);
        tokenizer.EosId.Should().Be(32000);
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
