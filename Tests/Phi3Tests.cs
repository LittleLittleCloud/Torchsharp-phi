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

namespace Phi.Tests;

public class Phi3Tests
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var clipTextModel = Phi3ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16);
        var state_dict_str = clipTextModel.Peek();
        Approvals.Verify(state_dict_str);
    }
}
