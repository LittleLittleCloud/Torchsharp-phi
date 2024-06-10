using FluentAssertions;
using Phi.Module;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Phi.Tests;

public class PhiLint4LinearTests
{
    [Fact]
    public void SizeTests()
    {
        // meta is critical for the test
        // as the size of the model to test is 372 GB
        // and can't be loaded in real device like cpu or cuda
        var device = "meta";
        var model = new PhiInt4Linear(100000, 100, device: device);

        var sizeInBytes = model.GetSizeInBytes();

        var sizeInGigaBytes = sizeInBytes / 1024 / 1024;
        sizeInGigaBytes.Should().Be(38);

        // to int8
        model.Quantize();
        var sizeInBytesAfterInt8 = model.GetSizeInBytes();

        var sizeInGigaBytesAfterInt8 = sizeInBytesAfterInt8 / 1024 / 1024;
        sizeInGigaBytesAfterInt8.Should().Be(4);
    }

    [Fact]
    public void ForwardTest()
    {
        var device = "cpu";
        var model = new PhiInt4Linear(123, 10, device: device);

        // set both weight and bias to rand int8 values
        // and compare the result before and after ToInt8

        var input = torch.randint(-8, 7, [10 ,2200, 123], device: device);
        var weight = torch.randint(-8, 7, [10, 123], device: device);
        var bias = torch.randint(-8, 7, [10], device: device);

        var weightStr = weight.Peek("weight").ToString();

        weight = (weight + 8).view(-1).to(torch.uint8);
        var weightPlaceHolderDim = (int)weight.size(0);
        weightPlaceHolderDim = weightPlaceHolderDim / 2 + weightPlaceHolderDim % 2;
        var weightPlaceHolder = weight[..weightPlaceHolderDim];
        weightPlaceHolder = weightPlaceHolder * 16 + weight[weightPlaceHolderDim..];

        var high4Bit = weightPlaceHolder / 16;
        var low4Bit = weightPlaceHolder % 16;
        weight = torch.cat(new Tensor[] { high4Bit, low4Bit }).view(10, 123);
        weight = weight.to(torch.int64);
        weight -= 8;
        weight.Peek("weight").Should().Be(weightStr);

        // scale and zero point on vector-wise
        //input = input * (250 / (torch.max(input) - torch.min(input)));
        //weight = weight * (250 / (torch.max(weight) - torch.min(weight))).view(-1, 1);
        //bias = bias * (250 / (torch.max(bias) - torch.min(bias)));

        model.load_state_dict(new Dictionary<string, Tensor>
        {
            ["weight"] = weight,
            ["bias"] = bias
        });

        var resultBeforeInt8 = model.forward(input);

        model.Quantize();

        var resultAfterInt8 = model.forward(input);

        // compare the result
        resultBeforeInt8.Peek("result").Should().Be(resultAfterInt8.Peek("result"));
    }
}
