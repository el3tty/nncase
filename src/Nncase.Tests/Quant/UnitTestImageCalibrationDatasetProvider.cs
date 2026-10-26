// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Quantization;
using SkiaSharp;
using Xunit;

namespace Nncase.Tests.QuantTest;

/// <summary>
/// Checks resize mode and channel order of <see cref="ImageCalibrationDatasetProvider"/> on generated lossless images.
/// The model input is 32x32 float with input range 0..255, so tensor values are the raw 0..255 pixel values.
/// </summary>
public sealed class UnitTestImageCalibrationDatasetProvider : IDisposable
{
    private const int Size = 32;

    private const float Pad = 114;

    private static readonly float[] Red = { 255, 0, 0 };

    private static readonly float[] Green = { 0, 255, 0 };

    private static readonly float[] Blue = { 0, 0, 255 };

    private static readonly float[] Gray = { Pad, Pad, Pad };

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "nncase_image_calib_" + Guid.NewGuid().ToString("N"));

    public UnitTestImageCalibrationDatasetProvider()
    {
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        Directory.Delete(_dir, true);
    }

    [Fact]
    public async Task TestLetterboxIsCenteredByDefault()
    {
        // 64x48 -> 32x24, centered: rows 4..27 hold the image, 4 gray rows above and below.
        WriteSplitImage(64, 48);
        var t = await LoadAsync(new ImageCalibrationDatasetProvider(new[] { CreateInput() }, _dir, new[] { 0f, 255f }));

        Assert.Equal(Gray, Pixel(t, 16, 0));
        Assert.Equal(Gray, Pixel(t, 16, 3));
        Assert.Equal(Red, Pixel(t, 8, 4));
        Assert.Equal(Green, Pixel(t, 24, 16));
        Assert.Equal(Red, Pixel(t, 8, 27));
        Assert.Equal(Gray, Pixel(t, 16, 28));
        Assert.Equal(Gray, Pixel(t, 16, 31));
    }

    [Fact]
    public async Task TestLetterboxTopLeft()
    {
        // 64x48 -> 32x24 at the top-left: rows 0..23 hold the image, rows 24..31 are gray.
        WriteSplitImage(64, 48);
        var t = await LoadAsync(CalibImageResize.LetterboxTopLeft, false);

        Assert.Equal(Blue, Pixel(t, 1, 1));
        Assert.Equal(Red, Pixel(t, 8, 12));
        Assert.Equal(Green, Pixel(t, 24, 12));
        Assert.Equal(Red, Pixel(t, 8, 23));
        Assert.Equal(Gray, Pixel(t, 8, 24));
        Assert.Equal(Gray, Pixel(t, 31, 31));
    }

    [Fact]
    public async Task TestLetterboxTopLeftTruncatesSize()
    {
        // 100x30 scales by 0.32 to 32x9.6: Letterbox rounds to 10 rows (centered at 11..20),
        // LetterboxTopLeft truncates to 9 rows like YOLOX, so row 9 is already padding.
        WriteSolidImage(100, 30, Red);

        var centered = await LoadAsync(CalibImageResize.Letterbox, false);
        Assert.Equal(Gray, Pixel(centered, 16, 10));
        Assert.Equal(Red, Pixel(centered, 16, 11));
        Assert.Equal(Red, Pixel(centered, 16, 20));
        Assert.Equal(Gray, Pixel(centered, 16, 21));

        var topLeft = await LoadAsync(CalibImageResize.LetterboxTopLeft, false);
        Assert.Equal(Red, Pixel(topLeft, 16, 0));
        Assert.Equal(Red, Pixel(topLeft, 16, 8));
        Assert.Equal(Gray, Pixel(topLeft, 16, 9));
    }

    [Fact]
    public async Task TestStretch()
    {
        // 64x48 -> 32x32 without padding: the whole input is image.
        WriteSplitImage(64, 48);
        var t = await LoadAsync(CalibImageResize.Stretch, false);

        Assert.DoesNotContain(Enumerable.Range(0, Size * Size), i => Pixel(t, i % Size, i / Size).SequenceEqual(Gray));
        Assert.Equal(Blue, Pixel(t, 1, 1));
        Assert.Equal(Red, Pixel(t, 8, 0));
        Assert.Equal(Red, Pixel(t, 8, 31));
        Assert.Equal(Green, Pixel(t, 24, 31));
    }

    [Theory]
    [InlineData(CalibImageResize.Letterbox)]
    [InlineData(CalibImageResize.LetterboxTopLeft)]
    [InlineData(CalibImageResize.Stretch)]
    public async Task TestBGRSwapsChannelPlanes(CalibImageResize resize)
    {
        WriteSplitImage(64, 48);
        var rgb = await LoadAsync(resize, false);
        var bgr = await LoadAsync(resize, true);

        int plane = Size * Size;
        Assert.Equal(rgb[(2 * plane)..], bgr[..plane]);
        Assert.Equal(rgb[plane..(2 * plane)], bgr[plane..(2 * plane)]);
        Assert.Equal(rgb[..plane], bgr[(2 * plane)..]);
        Assert.Equal(Red, Pixel(rgb, 8, 12));
        Assert.Equal(Blue, Pixel(bgr, 8, 12));
    }

    [Theory]
    [InlineData(CalibImageResize.Letterbox)]
    [InlineData(CalibImageResize.LetterboxTopLeft)]
    [InlineData(CalibImageResize.Stretch)]
    public async Task TestImageAtInputSizeIsUsedAsIs(CalibImageResize resize)
    {
        WriteSplitImage(Size, Size);
        var t = await LoadAsync(resize, false);

        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                Assert.Equal(ExpectedSplitColor(x, y, Size, Size), Pixel(t, x, y));
            }
        }
    }

    private static Var CreateInput() => new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, Size, Size }));

    /// <summary>
    /// Gets the R, G, B (or B, G, R) values of pixel (x, y) from an NCHW 1x3xSizexSize tensor.
    /// </summary>
    private static float[] Pixel(float[] t, int x, int y)
    {
        int plane = Size * Size;
        int i = (y * Size) + x;
        return new[] { t[i], t[plane + i], t[(2 * plane) + i] };
    }

    private static float[] ExpectedSplitColor(int x, int y, int width, int height)
    {
        int block = width / 8;
        if (x < block && y < block)
        {
            return Blue;
        }

        return x < width / 2 ? Red : Green;
    }

    private static SKColor ToColor(float[] rgb) => new SKColor((byte)rgb[0], (byte)rgb[1], (byte)rgb[2]);

    private static async Task<float[]> LoadAsync(ImageCalibrationDatasetProvider provider)
    {
        Assert.Equal(1, provider.Count);
        await foreach (var sample in provider.Samples)
        {
            var tensor = sample.Values.Single().AsTensor();
            Assert.Equal(new[] { 1, 3, Size, Size }, tensor.Shape.ToValueArray());
            return tensor.ToArray<float>();
        }

        throw new InvalidOperationException("The provider returned no samples.");
    }

    private Task<float[]> LoadAsync(CalibImageResize resize, bool isBGR) =>
        LoadAsync(new ImageCalibrationDatasetProvider(new[] { CreateInput() }, _dir, new[] { 0f, 255f }, resize, isBGR));

    /// <summary>
    /// Writes a PNG with a red left half, a green right half and a blue block (1/8 of the width) at the top-left corner.
    /// </summary>
    private void WriteSplitImage(int width, int height)
    {
        WriteImage(width, height, (x, y) => ExpectedSplitColor(x, y, width, height));
    }

    private void WriteSolidImage(int width, int height, float[] rgb)
    {
        WriteImage(width, height, (x, y) => rgb);
    }

    private void WriteImage(int width, int height, Func<int, int, float[]> color)
    {
        using var bitmap = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Opaque);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                bitmap.SetPixel(x, y, ToColor(color(x, y)));
            }
        }

        using var data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(Path.Combine(_dir, "image.png"), data.ToArray());
    }
}
