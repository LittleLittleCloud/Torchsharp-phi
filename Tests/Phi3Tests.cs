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
using System.Text.Json;

namespace Phi.Tests;

public class Phi3Tests
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Phi3Mini4KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var model = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = model.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Phi3Medium4KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-medium-4k-instruct";
        var model = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = model.Peek();
        Approvals.Verify(state_dict_str);
    }


    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Phi3Medium128KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-medium-128k-instruct";
        var model = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = model.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    public async Task Phi3Medium128kModelSizeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-medium-128k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        torch.InitializeDevice(META);
        var model = new Phi3ForCasualLM(modelConfig);

        model = model.to("meta");
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Phi3Mini128KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-128k-instruct";
        var model = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = model.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Phi3Mini128KLayerSizeTest()
    {
        var dtype = ScalarType.BFloat16;
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-128k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = dtype;
        var model = new Phi3ForCasualLM(modelConfig);

        var size = model.GetSizeForEachDynamicLayerInBytes();
        // convert size to MB
        var sizeInMB = size.ToDictionary(x => x.Key, x => x.Value * 1.0f / 1024 / 1024);

        var json = JsonSerializer.Serialize(sizeInMB, new JsonSerializerOptions { WriteIndented = true });
        Approvals.Verify(json);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task TokenizerTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var tokenizer = LLama2Tokenizer.FromPretrained(modelWeightFolder);
        tokenizer.BosId.Should().Be(1);
        tokenizer.EosId.Should().Be(2);
        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?",
            "\nCount to 3\n",
            "<|user|>\nCount to 3<|end|>\n<|assistant|>",
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
