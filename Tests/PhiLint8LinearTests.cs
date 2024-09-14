using FluentAssertions;
using Phi.Module;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace Phi.Tests;

public class PhiLint8LinearTests
{
    private ITestOutputHelper output;

    public PhiLint8LinearTests(ITestOutputHelper output)
    {
        this.output = output;
    }

    [Fact]
    public void SizeTests()
    {
        // meta is critical for the test
        // as the size of the model to test is 372 GB
        // and can't be loaded in real device like cpu or cuda
        var device = "meta";
        var model = new PhiInt8Linear(100000, 100, device: device);

        var sizeInBytes = model.GetSizeInBytes();

        var sizeInGigaBytes = sizeInBytes / 1024 / 1024;
        sizeInGigaBytes.Should().Be(38);

        // to int8
        model.Quantize();
        var sizeInBytesAfterInt8 = model.GetSizeInBytes();

        var sizeInGigaBytesAfterInt8 = sizeInBytesAfterInt8 / 1024 / 1024;
        sizeInGigaBytesAfterInt8.Should().Be(9);
    }

    [Fact]
    public void ForwardTest()
    {
        var device = "cpu";
        var model = new PhiInt8Linear(123, 10, device: device);

        // set both weight and bias to rand int8 values
        // and compare the result before and after ToInt8

        var input = torch.randint(-128, 127, [10, 2200, 123], device: device);
        var weight = torch.randint(-128, 127, [10, 123], device: device);
        var bias = torch.randint(-128, 127, [10], device: device);

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

    [Fact]
    public void MatMulitBenchmark()
    {
        var sizeX = new long[] { 1, 1000, 1000 };
        var sizeY = new long[] { 1000, 100 };
        var device = "cpu";
        // float32
        var x = torch.randn(sizeX, device: device);
        var y = torch.randn(sizeY, device: device);

        // warm up
        for (var i = 0; i < 10; i++)
        {
            var _ = torch.matmul(x, y);
        }

        // measure
        var timer = System.Diagnostics.Stopwatch.StartNew();
        for (var i = 0; i < 10; i++)
        {
            var _ = torch.matmul(x, y);
        }
        timer.Stop();

        output.WriteLine($"MatMulitBenchmark elapsed time: {timer.ElapsedMilliseconds} ms");

        // int8
        var xInt8 = x.to(ScalarType.Int8);
        var yInt8 = y.to(ScalarType.Int8);

        // warm up
        for (var i = 0; i < 10; i++)
        {
            var _ = torch.matmul(xInt8, yInt8);
        }

        // measure
        timer.Restart();
        for (var i = 0; i < 10; i++)
        {
            var _ = torch.matmul(xInt8, yInt8);
        }

        timer.Stop();

        output.WriteLine($"MatMulitBenchmark int8 elapsed time: {timer.ElapsedMilliseconds} ms");

    }
}
