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
}
