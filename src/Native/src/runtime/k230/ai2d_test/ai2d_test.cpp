// ai2d_test.cpp
//
// General affine transform of a 24-bit BMP on the K230 AI2D (rotate, scale,
// shear, flip, translate, or a raw 2x3 matrix), with an optional CPU
// reference renderer for comparing against the board.
//
//   usage: ai2d_test in.bmp out.bmp [transforms] [options]
//
// Transforms are applied in the order given, about the image centre:
//   --rotate DEG          clockwise on screen for positive DEG
//   --scale S | SX,SY     uniform or per-axis scale
//   --shear SHX,SHY       x' = x + SHX*y ; y' = y + SHY*x
//   --translate TX,TY     pixels, +x right, +y down
//   --flipx / --flipy     mirror left-right / top-bottom
//   --matrix a,b,tx,c,d,ty   raw FORWARD map (in -> out) in pixel-index
//                         coordinates: x' = a*x + b*y + tx ; y' = c*x + d*y + ty
//                         (cannot be combined with the transforms above)
//
// Stages (fixed order, whatever the order on the command line):
//   crop -> resize -> transforms above -> pad
//   --crop X,Y,W,H        take this window of the source first; the transforms,
//                         --matrix and --translate then work in window coordinates
//   --resize W,H          resample the (cropped) image to W x H before the transforms
//   --pad L,T,R,B | N     add a border around the final canvas (N = all four sides)
//   --pad-value V | R,G,B border colour, default 0
// Any of crop, resize, pad and --colorspace can be used on their own, without a transform.
//
// Options:
//   --interp bilinear|nearest        (default bilinear)
//   --colorspace NV12|NV21|I420      round-trip the image through a YUV 4:2:0 format
//                         (default: none, RGB in -> AI2D -> RGB out)
//   --cpu                    render in software instead of AI2D (reference output)
//   --dry-run                print matrix and canvas, write nothing
//
// Regression suite (see TestSuite below for the table of invocations):
//   --run-test-suite         run every invocation on ai2d_test.bmp (416x640 expected,
//                            128x128..640x640 accepted), write ai2d_test_output<N>.bmp,
//                            hash each output with XXH3-64 and compare with the line
//                            N of ai2d_test.hash (SUCCESS / FAIL per test)
//   --run-test-suite -f      do not verify: print the hashes and write ai2d_test.hash
//   --run-test-suite --cpu   run the suite in software; uses ai2d_test_cpu.hash so the
//                            board's ai2d_test.hash is never touched
// Generate the reference hashes once with -f on a known-good board, then run
// without -f on every board that has to match.
//
// The output canvas is the bounding box of the transformed image, re-centred.
// With --translate or --matrix it keeps the size of the image entering the
// transforms (after crop and resize) instead, since a re-centred canvas would
// cancel the translation. The pad border is added outside that canvas.
//
// On AI2D the stages map onto the hardware ones: crop_param for --crop;
// resize_param when --resize is used without any transform; and when --resize is
// combined with transforms, the scaling is folded into the affine matrix (AI2D
// refuses resize and affine in the same pass); pad_param for --pad.
//
// --colorspace: the RGB bitmap is converted to the chosen YUV 4:2:0 layout on the
// host (BT.601 full range, chroma averaged over each 2x2 block), AI2D reads that
// YUV buffer (src_format = NV12/NV21/I420) and writes planar RGB (the hardware
// colour-space converter does the way back), and the result is saved as BMP.
// With --cpu the same round trip is done in software before the warp. Width
// and height must be even.
//
// The AI2D affine boundary registers are fixed to the values the upstream
// callers pass: bound_val 127, bound_ind 0, bound_smooth 1; cord_round is 0
// for bilinear and 2 for nearest.
//
// Needs cxxopts (https://github.com/jarro2783/cxxopts) and xxHash
// (https://github.com/Cyan4973/xxHash, the single header xxhash.h) on the include path.
// Build on the board inside the face_detect project (same include paths / link
// line as utils_face_detect.cc), run as root: ai2d_builder opens /dev/ai2d and
// /dev/mem. Build with -DAI2D_STANDALONE to get only the CPU path anywhere.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cxxopts.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

#ifndef AI2D_STANDALONE
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/functional/ai2d/ai2d_builder.h>
#endif

// ===========================================================================
// Declarations
// ===========================================================================

// 2D affine transform stored as a 3x3 homogeneous matrix (bottom row 0 0 1).
// Coordinates are pixel indices: (0,0) is the centre of the top-left pixel,
// +x is right and +y is down.
class Mat3
{
public:
    Mat3(); // identity

    static Mat3 affine(double a, double b, double tx, double c, double d, double ty);
    static Mat3 translation(double tx, double ty);
    static Mat3 rotation(double degrees);      // clockwise on screen
    static Mat3 scaling(double sx, double sy);
    static Mat3 shear(double shx, double shy); // x' = x + shx*y, y' = y + shy*x
    static Mat3 flipX();                       // mirror left-right about x = 0
    static Mat3 flipY();                       // mirror top-bottom about y = 0

    Mat3 operator*(const Mat3 &rhs) const;     // rhs is applied first
    Mat3 then(const Mat3 &next) const;         // *this first, then next
    Mat3 inverse() const;                      // throws std::runtime_error if singular
    double determinant() const;                // of the 2x2 linear part
    bool isSingular() const;
    void apply(double x, double y, double &ox, double &oy) const;
    double at(int row, int col) const;
    std::vector<float> params() const;         // {a, b, tx, c, d, ty} as AI2D takes them

private:
    double m_[3][3];
};

// 24-bit BMP held as planar 8-bit R, G, B planes, top row first: the NCHW
// layout AI2D wants.
class Bitmap
{
public:
    Bitmap();
    Bitmap(int width, int height, uint8_t fill);

    void load(const std::string &path);       // uncompressed 24-bpp BMP; throws std::runtime_error
    void save(const std::string &path) const; // writes a bottom-up 24-bpp BMP; throws std::runtime_error

    int width() const;
    int height() const;
    size_t planeSize() const;                 // width * height
    size_t size() const;                      // 3 * planeSize()
    uint8_t *data();
    const uint8_t *data() const;
    uint8_t *plane(int channel);              // 0 = R, 1 = G, 2 = B
    const uint8_t *plane(int channel) const;

    Bitmap crop(int x, int y, int w, int h) const;                          // window must lie inside the image
    Bitmap padded(int left, int top, int right, int bottom, const uint8_t value[3]) const;

private:
    static uint16_t rd16(const uint8_t *p);
    static uint32_t rd32(const uint8_t *p);
    static void wr16(uint8_t *p, uint16_t v);
    static void wr32(uint8_t *p, uint32_t v);

    int w_;
    int h_;
    std::vector<uint8_t> data_;
};

// YUV 4:2:0 conversion of a Bitmap (BT.601 full range, chroma averaged over
// each 2x2 block, so width and height must be even).
//   NV12: Y plane, then interleaved U V
//   NV21: Y plane, then interleaved V U
//   I420: Y plane, then U plane, then V plane
class Yuv420
{
public:
    enum class Format { NV12, NV21, I420 };

    static bool parse(const std::string &name, Format &out); // case-insensitive
    static const char *name(Format f);
    static size_t bytes(int width, int height);               // width * height * 3 / 2
    static std::vector<uint8_t> encode(const Bitmap &rgb, Format f); // throws on odd size
    static Bitmap decode(const uint8_t *yuv, int width, int height, Format f);
};

// What the command line asks for. Transforms are composed in the order they
// appear on the command line.
class CommandLine
{
public:
    std::string input;
    std::string output;
    Mat3 centred;               // transforms composed in order, about the image centre
    Mat3 raw;                   // --matrix, in pixel-index coordinates
    bool hasOps = false;
    bool hasTranslate = false;
    bool useMatrix = false;
    bool nearest = false;
    bool cpu = false;
    bool dryRun = false;
    bool useYuv = false;                              // --colorspace given
    Yuv420::Format yuv = Yuv420::Format::NV12;

    bool runSuite = false;                            // --run-test-suite
    bool writeHashes = false;                         // -f: write ai2d_test.hash instead of verifying

    bool useCrop = false;                             // --crop
    int cropX = 0, cropY = 0, cropW = 0, cropH = 0;
    bool useResize = false;                           // --resize
    int resizeW = 0, resizeH = 0;
    bool usePad = false;                              // --pad
    int padL = 0, padT = 0, padR = 0, padB = 0;
    uint8_t padValue[3] = { 0, 0, 0 };                // --pad-value, per channel R, G, B

    // Returns false when the program should exit now (help shown or bad
    // arguments); exitCode says with which status.
    static bool parse(int argc, char **argv, CommandLine &out, int &exitCode);

    // True when the output keeps the input size (translate or raw matrix);
    // otherwise the canvas is the bounding box of the transformed image.
    bool keepsInputSize() const;

private:
    void compose(const Mat3 &op);
    static std::vector<double> numbers(const std::string &text, size_t minCount, size_t maxCount, const std::string &usage);
    static std::vector<int> integers(const std::string &text, size_t minCount, size_t maxCount, const std::string &usage);
};

// Canvas size plus the final forward matrix (input pixel index -> output pixel
// index), and the means to render with them.
class Plan
{
public:
    void build(const CommandLine &cmd, int srcWidth, int srcHeight); // throws std::runtime_error
    Bitmap run(const Bitmap &src) const;                             // AI2D, or the CPU reference with --cpu
    void describe(std::ostream &os) const;

private:
    Bitmap runCpu(const Bitmap &src) const;
#ifndef AI2D_STANDALONE
    Bitmap runAi2d(const Bitmap &src) const;
#endif

    // How one AI2D pass realises the plan.
    enum class Mode
    {
        Copy,   // crop and/or pad only
        Resize, // resize_param (no transform)
        Affine  // affine_param; --resize, if any, is folded into the matrix
    };

    int srcW_ = 0;
    int srcH_ = 0;
    bool useCrop_ = false;
    int cropX_ = 0, cropY_ = 0, cropW_ = 0, cropH_ = 0;   // window of the source (the whole image without --crop)
    bool useResize_ = false;
    int resizeW_ = 0, resizeH_ = 0;
    bool usePad_ = false;
    int padL_ = 0, padT_ = 0, padR_ = 0, padB_ = 0;
    uint8_t padValue_[3] = { 0, 0, 0 };
    Mode mode_ = Mode::Affine;
    int outW_ = 0;                                         // canvas before padding
    int outH_ = 0;
    int totalW_ = 0;                                       // canvas including the pad border
    int totalH_ = 0;
    Mat3 forward_;
    bool nearest_ = false;
    bool cpu_ = false;
    bool useYuv_ = false;
    Yuv420::Format yuv_ = Yuv420::Format::NV12;
    std::string canvasDesc_;
};

// The regression suite behind --run-test-suite: a fixed table of command lines
// (options only, no file names) run on ai2d_test.bmp, one output BMP and one
// XXH3-64 hash per entry.
class TestSuite
{
public:
    // Returns the process exit code: 0 when every test passed (or the hash file
    // was written), 1 otherwise. Set writeHashes for -f, cpu to run in software.
    static int run(bool writeHashes, bool cpu);

    static size_t count();

private:
    static std::string describe(size_t index);                           // the options of one entry, as a string
    static std::string outputName(size_t index);                         // ai2d_test_output<N>.bmp
    static std::string hashFile(const std::string &path);                // XXH3-64 of the file's bytes, 16 hex digits
    static std::vector<std::string> readHashes(const std::string &path); // one hash per line; throws if unreadable
};

// ===========================================================================
// Constants
// ===========================================================================

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kSingularEps = 1e-9;
constexpr int kMaxDim = 16383;                       // dst_height_shape is a 14-bit field
constexpr double kMaxOutputBytes = 512.0 * 1024 * 1024;

// AI2D affine boundary registers, fixed to what the upstream callers pass.
constexpr int kBoundVal = 127;                       // fill for samples outside the source (semantics unconfirmed)
constexpr int kBoundInd = 0;
constexpr int kBoundSmooth = 1;
} // namespace

// ===========================================================================
// Mat3
// ===========================================================================

Mat3::Mat3()
    : m_{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
{
}

Mat3 Mat3::affine(double a, double b, double tx, double c, double d, double ty)
{
    Mat3 r;
    r.m_[0][0] = a; r.m_[0][1] = b; r.m_[0][2] = tx;
    r.m_[1][0] = c; r.m_[1][1] = d; r.m_[1][2] = ty;
    return r;
}

Mat3 Mat3::translation(double tx, double ty)
{
    return affine(1, 0, tx, 0, 1, ty);
}

Mat3 Mat3::rotation(double degrees)
{
    // y points down, so this matrix turns the image clockwise as seen on screen
    const double th = degrees * kPi / 180.0;
    return affine(std::cos(th), -std::sin(th), 0, std::sin(th), std::cos(th), 0);
}

Mat3 Mat3::scaling(double sx, double sy)
{
    return affine(sx, 0, 0, 0, sy, 0);
}

Mat3 Mat3::shear(double shx, double shy)
{
    return affine(1, shx, 0, shy, 1, 0);
}

Mat3 Mat3::flipX()
{
    return affine(-1, 0, 0, 0, 1, 0);
}

Mat3 Mat3::flipY()
{
    return affine(1, 0, 0, 0, -1, 0);
}

Mat3 Mat3::operator*(const Mat3 &rhs) const
{
    Mat3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            double s = 0;
            for (int k = 0; k < 3; k++)
                s += m_[i][k] * rhs.m_[k][j];
            r.m_[i][j] = s;
        }
    return r;
}

Mat3 Mat3::then(const Mat3 &next) const
{
    return next * *this;
}

double Mat3::determinant() const
{
    return m_[0][0] * m_[1][1] - m_[0][1] * m_[1][0];
}

bool Mat3::isSingular() const
{
    return std::fabs(determinant()) < kSingularEps;
}

Mat3 Mat3::inverse() const
{
    if (isSingular())
        throw std::runtime_error("transform is singular (zero determinant), cannot be inverted");

    // x = A^-1 * (y - t)  =>  inverse linear part A^-1, inverse translation -A^-1 * t
    const double det = determinant();
    const double ia = m_[1][1] / det, ib = -m_[0][1] / det;
    const double ic = -m_[1][0] / det, id = m_[0][0] / det;
    return affine(ia, ib, -(ia * m_[0][2] + ib * m_[1][2]),
                  ic, id, -(ic * m_[0][2] + id * m_[1][2]));
}

void Mat3::apply(double x, double y, double &ox, double &oy) const
{
    ox = m_[0][0] * x + m_[0][1] * y + m_[0][2];
    oy = m_[1][0] * x + m_[1][1] * y + m_[1][2];
}

double Mat3::at(int row, int col) const
{
    return m_[row][col];
}

std::vector<float> Mat3::params() const
{
    return { static_cast<float>(m_[0][0]), static_cast<float>(m_[0][1]), static_cast<float>(m_[0][2]),
             static_cast<float>(m_[1][0]), static_cast<float>(m_[1][1]), static_cast<float>(m_[1][2]) };
}

// ===========================================================================
// Bitmap
// ===========================================================================

Bitmap::Bitmap()
    : w_(0)
    , h_(0)
{
}

Bitmap::Bitmap(int width, int height, uint8_t fill)
    : w_(width)
    , h_(height)
    , data_(3 * static_cast<size_t>(width) * static_cast<size_t>(height), fill)
{
}

uint16_t Bitmap::rd16(const uint8_t *p)
{
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t Bitmap::rd32(const uint8_t *p)
{
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void Bitmap::wr16(uint8_t *p, uint16_t v)
{
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
}

void Bitmap::wr32(uint8_t *p, uint32_t v)
{
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
    p[2] = (v >> 16) & 0xFF;
    p[3] = (v >> 24) & 0xFF;
}

void Bitmap::load(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    const std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if (file.size() < 54 || file[0] != 'B' || file[1] != 'M')
        throw std::runtime_error("not a BMP file");

    const uint32_t off = rd32(&file[10]);
    const uint32_t hdr = rd32(&file[14]);
    const int32_t w = static_cast<int32_t>(rd32(&file[18]));
    int32_t h = static_cast<int32_t>(rd32(&file[22]));
    const uint16_t bpp = rd16(&file[28]);
    const uint32_t comp = rd32(&file[30]);

    if (hdr < 40 || bpp != 24 || comp != 0)
        throw std::runtime_error("only uncompressed 24-bpp BMP (BITMAPINFOHEADER) is supported");

    const bool bottomUp = h > 0; // positive height = rows stored bottom-to-top
    if (h == std::numeric_limits<int32_t>::min())
        throw std::runtime_error("bad BMP size");
    if (h < 0)
        h = -h;
    if (w <= 0 || h <= 0)
        throw std::runtime_error("bad BMP size");

    const size_t stride = ((static_cast<size_t>(w) * 3 + 3) & ~static_cast<size_t>(3)); // rows padded to 4 bytes
    if (static_cast<size_t>(off) + stride * static_cast<size_t>(h) > file.size())
        throw std::runtime_error("BMP pixel data truncated");

    const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> planar(3 * plane, 0);

    for (int y = 0; y < h; y++)
    {
        const int srcRow = bottomUp ? (h - 1 - y) : y;
        const uint8_t *s = &file[off + static_cast<size_t>(srcRow) * stride];
        for (int x = 0; x < w; x++)
        {
            // BMP stores B, G, R
            planar[0 * plane + static_cast<size_t>(y) * w + x] = s[3 * x + 2]; // R
            planar[1 * plane + static_cast<size_t>(y) * w + x] = s[3 * x + 1]; // G
            planar[2 * plane + static_cast<size_t>(y) * w + x] = s[3 * x + 0]; // B
        }
    }

    w_ = w;
    h_ = h;
    data_.swap(planar);
}

void Bitmap::save(const std::string &path) const
{
    if (w_ <= 0 || h_ <= 0)
        throw std::runtime_error("cannot save an empty image");

    const size_t stride = ((static_cast<size_t>(w_) * 3 + 3) & ~static_cast<size_t>(3));
    const size_t plane = planeSize();
    const uint32_t pixBytes = static_cast<uint32_t>(stride * static_cast<size_t>(h_));

    std::vector<uint8_t> out(54 + pixBytes, 0);
    out[0] = 'B';
    out[1] = 'M';
    wr32(&out[2], 54 + pixBytes);
    wr32(&out[10], 54);   // pixel data offset
    wr32(&out[14], 40);   // BITMAPINFOHEADER size
    wr32(&out[18], static_cast<uint32_t>(w_));
    wr32(&out[22], static_cast<uint32_t>(h_)); // positive = bottom-up
    wr16(&out[26], 1);
    wr16(&out[28], 24);
    wr32(&out[30], 0);
    wr32(&out[34], pixBytes);
    wr32(&out[38], 2835);
    wr32(&out[42], 2835);

    for (int y = 0; y < h_; y++)
    {
        uint8_t *d = &out[54 + static_cast<size_t>(h_ - 1 - y) * stride]; // file row 0 = bottom image row
        for (int x = 0; x < w_; x++)
        {
            d[3 * x + 0] = data_[2 * plane + static_cast<size_t>(y) * w_ + x]; // B
            d[3 * x + 1] = data_[1 * plane + static_cast<size_t>(y) * w_ + x]; // G
            d[3 * x + 2] = data_[0 * plane + static_cast<size_t>(y) * w_ + x]; // R
        }
    }

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f)
        throw std::runtime_error("cannot write " + path);
}

int Bitmap::width() const
{
    return w_;
}

int Bitmap::height() const
{
    return h_;
}

size_t Bitmap::planeSize() const
{
    return static_cast<size_t>(w_) * static_cast<size_t>(h_);
}

size_t Bitmap::size() const
{
    return data_.size();
}

uint8_t *Bitmap::data()
{
    return data_.data();
}

const uint8_t *Bitmap::data() const
{
    return data_.data();
}

uint8_t *Bitmap::plane(int channel)
{
    return data_.data() + static_cast<size_t>(channel) * planeSize();
}

const uint8_t *Bitmap::plane(int channel) const
{
    return data_.data() + static_cast<size_t>(channel) * planeSize();
}

Bitmap Bitmap::crop(int x, int y, int w, int h) const
{
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > w_ || y + h > h_)
        throw std::runtime_error("crop window outside the image");
    Bitmap out(w, h, 0);
    for (int ch = 0; ch < 3; ch++)
        for (int row = 0; row < h; row++)
            std::memcpy(out.plane(ch) + static_cast<size_t>(row) * w,
                        plane(ch) + static_cast<size_t>(y + row) * w_ + x, static_cast<size_t>(w));
    return out;
}

Bitmap Bitmap::padded(int left, int top, int right, int bottom, const uint8_t value[3]) const
{
    const int nw = w_ + left + right, nh = h_ + top + bottom;
    Bitmap out(nw, nh, 0);
    for (int ch = 0; ch < 3; ch++)
    {
        std::fill(out.plane(ch), out.plane(ch) + out.planeSize(), value[ch]);
        for (int row = 0; row < h_; row++)
            std::memcpy(out.plane(ch) + static_cast<size_t>(top + row) * nw + left,
                        plane(ch) + static_cast<size_t>(row) * w_, static_cast<size_t>(w_));
    }
    return out;
}

// ===========================================================================
// Yuv420
// ===========================================================================

namespace
{
uint8_t toByte(double v)
{
    return static_cast<uint8_t>(std::min(255.0, std::max(0.0, std::floor(v + 0.5))));
}
} // namespace

bool Yuv420::parse(const std::string &text, Format &out)
{
    std::string t;
    for (char c : text)
        t += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (t == "NV12") out = Format::NV12;
    else if (t == "NV21") out = Format::NV21;
    else if (t == "I420") out = Format::I420;
    else return false;
    return true;
}

const char *Yuv420::name(Format f)
{
    switch (f)
    {
    case Format::NV12: return "NV12";
    case Format::NV21: return "NV21";
    default: return "I420";
    }
}

size_t Yuv420::bytes(int width, int height)
{
    return static_cast<size_t>(width) * height * 3 / 2;
}

std::vector<uint8_t> Yuv420::encode(const Bitmap &rgb, Format f)
{
    const int w = rgb.width(), h = rgb.height();
    if (w % 2 != 0 || h % 2 != 0)
        throw std::runtime_error("--colorspace needs an even width and height (image is " + std::to_string(w) + "x" + std::to_string(h) + ")");

    const size_t plane = rgb.planeSize();
    const uint8_t *R = rgb.plane(0), *G = rgb.plane(1), *B = rgb.plane(2);
    std::vector<uint8_t> out(bytes(w, h));

    // luma per pixel
    for (size_t i = 0; i < plane; i++)
        out[i] = toByte(0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i]);

    // chroma per pixel, averaged over each 2x2 block
    const int cw = w / 2, ch = h / 2;
    uint8_t *uBase = &out[plane];
    uint8_t *vBase = (f == Format::I420) ? &out[plane + (size_t)cw * ch] : nullptr;
    for (int by = 0; by < ch; by++)
    {
        for (int bx = 0; bx < cw; bx++)
        {
            double u = 0, v = 0;
            for (int dy = 0; dy < 2; dy++)
            {
                for (int dx = 0; dx < 2; dx++)
                {
                    const size_t i = (size_t)(2 * by + dy) * w + (2 * bx + dx);
                    u += -0.168736 * R[i] - 0.331264 * G[i] + 0.5 * B[i] + 128.0;
                    v += 0.5 * R[i] - 0.418688 * G[i] - 0.081312 * B[i] + 128.0;
                }
            }
            const uint8_t U = toByte(u / 4), V = toByte(v / 4);
            const size_t c = (size_t)by * cw + bx;
            if (f == Format::I420)
            {
                uBase[c] = U;
                vBase[c] = V;
            }
            else
            {
                uBase[2 * c + 0] = (f == Format::NV12) ? U : V;
                uBase[2 * c + 1] = (f == Format::NV12) ? V : U;
            }
        }
    }
    return out;
}

Bitmap Yuv420::decode(const uint8_t *yuv, int w, int h, Format f)
{
    Bitmap rgb(w, h, 0);
    const size_t plane = rgb.planeSize();
    const int cw = w / 2;
    const uint8_t *uBase = yuv + plane;
    const uint8_t *vBase = (f == Format::I420) ? yuv + plane + (size_t)cw * (h / 2) : nullptr;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            const size_t c = (size_t)(y / 2) * cw + (x / 2); // chroma replicated over the 2x2 block
            double U, V;
            if (f == Format::I420)
            {
                U = uBase[c];
                V = vBase[c];
            }
            else
            {
                U = uBase[2 * c + (f == Format::NV12 ? 0 : 1)];
                V = uBase[2 * c + (f == Format::NV12 ? 1 : 0)];
            }
            const double Y = yuv[(size_t)y * w + x];
            const size_t i = (size_t)y * w + x;
            rgb.plane(0)[i] = toByte(Y + 1.402 * (V - 128.0));
            rgb.plane(1)[i] = toByte(Y - 0.344136 * (U - 128.0) - 0.714136 * (V - 128.0));
            rgb.plane(2)[i] = toByte(Y + 1.772 * (U - 128.0));
        }
    }
    return rgb;
}

// ===========================================================================
// CommandLine
// ===========================================================================

std::vector<int> CommandLine::integers(const std::string &text, size_t minCount, size_t maxCount, const std::string &usage)
{
    std::vector<int> out;
    for (double d : numbers(text, minCount, maxCount, usage))
    {
        if (d != std::floor(d) || std::fabs(d) > 1e9)
            throw std::runtime_error(usage + " (whole numbers)");
        out.push_back(static_cast<int>(d));
    }
    return out;
}

void CommandLine::compose(const Mat3 &op)
{
    centred = centred.then(op);
    hasOps = true;
}

bool CommandLine::keepsInputSize() const
{
    return hasTranslate || useMatrix;
}

// Strict comma-separated list of finite numbers; throws `usage` on any problem.
std::vector<double> CommandLine::numbers(const std::string &text, size_t minCount, size_t maxCount, const std::string &usage)
{
    std::vector<double> v;
    size_t pos = 0;
    for (;;)
    {
        const size_t comma = text.find(',', pos);
        const std::string tok = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        char *end = nullptr;
        const double d = std::strtod(tok.c_str(), &end);
        if (tok.empty() || end == tok.c_str() || *end != '\0' || !std::isfinite(d))
            throw std::runtime_error(usage);
        v.push_back(d);
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    if (v.size() < minCount || v.size() > maxCount)
        throw std::runtime_error(usage);
    return v;
}

bool CommandLine::parse(int argc, char **argv, CommandLine &out, int &exitCode)
{
    cxxopts::Options cli("ai2d_test",
                         "General affine transform of a 24-bit BMP on the K230 AI2D (or on the CPU with --cpu).\n"
                         "Transforms are applied in the order given, about the image centre.");
    cli.positional_help("in.bmp out.bmp").show_positional_help();

    cli.add_options()
        ("h,help", "Print this help")
        ("run-test-suite", "Run the built-in regression suite on ai2d_test.bmp and verify the output hashes against ai2d_test.hash")
        ("f", "With --run-test-suite: write ai2d_test.hash from this run instead of verifying");
    cli.add_options("Transforms")
        ("rotate", "Rotate clockwise by DEG degrees", cxxopts::value<std::string>(), "DEG")
        ("scale", "Scale uniformly by S, or by SX,SY", cxxopts::value<std::string>(), "S|SX,SY")
        ("shear", "x' = x + SHX*y, y' = y + SHY*x", cxxopts::value<std::string>(), "SHX,SHY")
        ("translate", "Move by TX,TY pixels (+x right, +y down); the output keeps the input size",
         cxxopts::value<std::string>(), "TX,TY")
        ("flipx", "Mirror left-right")
        ("flipy", "Mirror top-bottom")
        ("matrix", "Raw forward map x' = a*x + b*y + tx, y' = c*x + d*y + ty in pixel-index coordinates; "
                   "the output keeps the input size; cannot be combined with the transforms above",
         cxxopts::value<std::string>(), "a,b,tx,c,d,ty");
    cli.add_options("Stages")
        ("crop", "Take the window at X,Y of size W x H from the source first", cxxopts::value<std::string>(), "X,Y,W,H")
        ("resize", "Resample to W x H before the transforms", cxxopts::value<std::string>(), "W,H")
        ("pad", "Add a border around the result: N on every side, or L,T,R,B", cxxopts::value<std::string>(), "N|L,T,R,B")
        ("pad-value", "Border colour (default 0)", cxxopts::value<std::string>(), "V|R,G,B");
    cli.add_options("Rendering")
        ("interp", "Sampling: bilinear or nearest", cxxopts::value<std::string>()->default_value("bilinear"), "MODE")
        ("colorspace", "Round-trip through a YUV 4:2:0 format (default: none, stay RGB)",
         cxxopts::value<std::string>(), "NV12|NV21|I420")
        ("cpu", "Render in software instead of on AI2D (reference output)")
        ("dry-run", "Print the matrix and canvas, write nothing");
    cli.add_options("Positional")
        ("input", "Input BMP", cxxopts::value<std::string>())
        ("output", "Output BMP", cxxopts::value<std::string>());
    cli.parse_positional({ "input", "output" });

    const auto helpText = [&cli]() {
        return cli.help({ "", "Transforms", "Stages", "Rendering" })
            + "\nExamples:\n"
              "  ai2d_test in.bmp out.bmp --rotate 45\n"
              "  ai2d_test in.bmp out.bmp --rotate 30 --scale 1.5 --shear 0.2,0\n"
              "  ai2d_test in.bmp out.bmp --crop 100,50,320,240 --resize 224,224 --pad 8 --pad-value 114,114,114\n"
              "  ai2d_test --run-test-suite -f      # once, on a known-good board: record hashes\n"
              "  ai2d_test --run-test-suite         # afterwards: verify them\n"
              "  ai2d_test in.bmp out.bmp --matrix 0.9,0.2,10,-0.1,1.1,5 --cpu\n";
    };

    try
    {
        if (argc == 1)
        {
            std::cerr << helpText();
            exitCode = 1;
            return false;
        }

        const cxxopts::ParseResult r = cli.parse(argc, argv);
        if (r.count("help"))
        {
            std::cout << helpText();
            exitCode = 0;
            return false;
        }
        if (r.count("run-test-suite"))
        {
            for (const cxxopts::KeyValue &kv : r.arguments())
                if (kv.key() != "run-test-suite" && kv.key() != "f" && kv.key() != "cpu")
                    throw std::runtime_error("--run-test-suite only combines with -f and --cpu (it has its own files and options)");
            if (!r.unmatched().empty())
                throw std::runtime_error("--run-test-suite takes no file names");
            out.runSuite = true;
            out.writeHashes = r.count("f") > 0;
            out.cpu = r["cpu"].as<bool>();
            exitCode = 0;
            return true;
        }
        if (r.count("f"))
            throw std::runtime_error("-f only makes sense with --run-test-suite");
        if (!r.count("input") || !r.count("output"))
            throw std::runtime_error("expected in.bmp and out.bmp");
        if (!r.unmatched().empty())
            throw std::runtime_error("unexpected argument '" + r.unmatched().front() + "' (transforms are options, e.g. --rotate 45)");

        out.input = r["input"].as<std::string>();
        out.output = r["output"].as<std::string>();

        // cxxopts hands back every occurrence in command-line order, which is
        // the order the transforms have to be composed in.
        for (const cxxopts::KeyValue &kv : r.arguments())
        {
            const std::string &key = kv.key();
            if (key == "rotate")
            {
                const auto v = numbers(kv.value(), 1, 1, "--rotate expects DEG");
                out.compose(Mat3::rotation(v[0]));
            }
            else if (key == "scale")
            {
                const auto v = numbers(kv.value(), 1, 2, "--scale expects S or SX,SY");
                out.compose(Mat3::scaling(v[0], v.size() == 2 ? v[1] : v[0]));
            }
            else if (key == "shear")
            {
                const auto v = numbers(kv.value(), 2, 2, "--shear expects SHX,SHY");
                out.compose(Mat3::shear(v[0], v[1]));
            }
            else if (key == "translate")
            {
                const auto v = numbers(kv.value(), 2, 2, "--translate expects TX,TY");
                out.compose(Mat3::translation(v[0], v[1]));
                out.hasTranslate = true;
            }
            else if (key == "flipx")
            {
                if (kv.as<bool>())
                    out.compose(Mat3::flipX());
            }
            else if (key == "flipy")
            {
                if (kv.as<bool>())
                    out.compose(Mat3::flipY());
            }
            else if (key == "matrix")
            {
                const auto v = numbers(kv.value(), 6, 6, "--matrix expects a,b,tx,c,d,ty");
                out.raw = Mat3::affine(v[0], v[1], v[2], v[3], v[4], v[5]);
                out.useMatrix = true;
            }
        }

        const std::string interp = r["interp"].as<std::string>();
        if (interp == "nearest")
            out.nearest = true;
        else if (interp != "bilinear")
            throw std::runtime_error("--interp expects bilinear or nearest");

        if (r.count("crop"))
        {
            const auto v = integers(r["crop"].as<std::string>(), 4, 4, "--crop expects X,Y,W,H");
            if (v[0] < 0 || v[1] < 0 || v[2] <= 0 || v[3] <= 0)
                throw std::runtime_error("--crop needs X,Y >= 0 and W,H > 0");
            out.useCrop = true;
            out.cropX = v[0];
            out.cropY = v[1];
            out.cropW = v[2];
            out.cropH = v[3];
        }
        if (r.count("resize"))
        {
            const auto v = integers(r["resize"].as<std::string>(), 2, 2, "--resize expects W,H");
            if (v[0] <= 0 || v[1] <= 0)
                throw std::runtime_error("--resize needs W,H > 0");
            out.useResize = true;
            out.resizeW = v[0];
            out.resizeH = v[1];
        }
        if (r.count("pad"))
        {
            const std::string text = r["pad"].as<std::string>();
            const auto v = integers(text, 1, 4, "--pad expects N or L,T,R,B");
            if (v.size() != 1 && v.size() != 4)
                throw std::runtime_error("--pad expects N or L,T,R,B");
            out.padL = v[0];
            out.padT = v.size() == 4 ? v[1] : v[0];
            out.padR = v.size() == 4 ? v[2] : v[0];
            out.padB = v.size() == 4 ? v[3] : v[0];
            if (out.padL < 0 || out.padT < 0 || out.padR < 0 || out.padB < 0)
                throw std::runtime_error("--pad values must be >= 0");
            out.usePad = out.padL + out.padT + out.padR + out.padB > 0; // all zero is no padding, as in the AI2D builder
        }
        if (r.count("pad-value"))
        {
            if (!r.count("pad"))
                throw std::runtime_error("--pad-value needs --pad");
            const auto v = integers(r["pad-value"].as<std::string>(), 1, 3, "--pad-value expects V or R,G,B");
            if (v.size() == 2)
                throw std::runtime_error("--pad-value expects V or R,G,B");
            for (int k = 0; k < 3; k++)
            {
                const int c = v.size() == 3 ? v[k] : v[0];
                if (c < 0 || c > 255)
                    throw std::runtime_error("--pad-value components must be 0..255");
                out.padValue[k] = static_cast<uint8_t>(c);
            }
        }

        if (r.count("colorspace"))
        {
            if (!Yuv420::parse(r["colorspace"].as<std::string>(), out.yuv))
                throw std::runtime_error("--colorspace expects NV12, NV21 or I420");
            out.useYuv = true;
        }

        out.cpu = r["cpu"].as<bool>();
        out.dryRun = r["dry-run"].as<bool>();

        if (out.useMatrix && out.hasOps)
            throw std::runtime_error("--matrix cannot be combined with --rotate/--scale/--shear/--translate/--flipx/--flipy");
        if (!out.useMatrix && !out.hasOps && !out.useCrop && !out.useResize && !out.usePad && !out.useYuv)
            throw std::runtime_error("no transform given");
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\nTry --help.\n";
        exitCode = 1;
        return false;
    }

    exitCode = 0;
    return true;
}

// ===========================================================================
// Plan
// ===========================================================================

void Plan::build(const CommandLine &cmd, int srcWidth, int srcHeight)
{
    srcW_ = srcWidth;
    srcH_ = srcHeight;
    nearest_ = cmd.nearest;
    cpu_ = cmd.cpu;
    useYuv_ = cmd.useYuv;
    yuv_ = cmd.yuv;
    if (useYuv_ && (srcW_ % 2 != 0 || srcH_ % 2 != 0))
        throw std::runtime_error("--colorspace needs an even width and height (image is " + std::to_string(srcW_) + "x" + std::to_string(srcH_) + ")");

    // ---- crop: the window of the source that everything else works on ----
    useCrop_ = cmd.useCrop;
    if (useCrop_)
    {
        if (static_cast<long long>(cmd.cropX) + cmd.cropW > srcW_ || static_cast<long long>(cmd.cropY) + cmd.cropH > srcH_)
            throw std::runtime_error("--crop window " + std::to_string(cmd.cropX) + "," + std::to_string(cmd.cropY) + " "
                                     + std::to_string(cmd.cropW) + "x" + std::to_string(cmd.cropH) + " lies outside the "
                                     + std::to_string(srcW_) + "x" + std::to_string(srcH_) + " image");
        if (useYuv_ && (cmd.cropX % 2 || cmd.cropY % 2 || cmd.cropW % 2 || cmd.cropH % 2))
            throw std::runtime_error("--crop with --colorspace needs even X,Y,W,H (chroma is shared by 2x2 pixels)");
        cropX_ = cmd.cropX;
        cropY_ = cmd.cropY;
        cropW_ = cmd.cropW;
        cropH_ = cmd.cropH;
    }
    else
    {
        cropX_ = cropY_ = 0;
        cropW_ = srcW_;
        cropH_ = srcH_;
    }

    // ---- resize: size of the image that enters the transforms ----
    useResize_ = cmd.useResize;
    resizeW_ = cmd.resizeW;
    resizeH_ = cmd.resizeH;
    const int entW = useResize_ ? resizeW_ : cropW_;
    const int entH = useResize_ ? resizeH_ : cropH_;
    if (entW > kMaxDim || entH > kMaxDim)
        throw std::runtime_error("--resize " + std::to_string(entW) + "x" + std::to_string(entH) + " exceeds the "
                                 + std::to_string(kMaxDim) + " limit");

    // cropped pixel index -> resized pixel index (half-pixel centres, like cv2.resize)
    Mat3 resizeMap;
    if (useResize_)
    {
        const double fx = static_cast<double>(entW) / cropW_, fy = static_cast<double>(entH) / cropH_;
        resizeMap = Mat3::affine(fx, 0, 0.5 * fx - 0.5, 0, fy, 0.5 * fy - 0.5);
    }

    const bool hasGeometry = cmd.useMatrix || cmd.hasOps;
    mode_ = hasGeometry ? Mode::Affine : (useResize_ ? Mode::Resize : Mode::Copy);

    // ---- pad ----
    usePad_ = cmd.usePad;
    padL_ = cmd.padL;
    padT_ = cmd.padT;
    padR_ = cmd.padR;
    padB_ = cmd.padB;
    std::memcpy(padValue_, cmd.padValue, sizeof padValue_);

    // transform about the centre of the image entering the transforms, in its
    // pixel-index coordinates
    const double cxi = (entW - 1) * 0.5, cyi = (entH - 1) * 0.5;
    const Mat3 geometry = cmd.useMatrix
        ? cmd.raw
        : Mat3::translation(-cxi, -cyi).then(cmd.centred).then(Mat3::translation(cxi, cyi));
    const Mat3 pre = resizeMap.then(geometry);

    if (pre.isSingular())
        throw std::runtime_error("transform is singular (zero determinant), cannot be inverted");

    // Canvas size is worked out in double and range-checked before it becomes an
    // int, so an absurd scale gives an error instead of overflowing.
    double sx = 0, sy = 0; // canvas placement shift
    double wd, hd;
    if (cmd.keepsInputSize())
    {
        wd = entW;
        hd = entH;
        canvasDesc_ = (useCrop_ || useResize_) ? "same as the cropped/resized image" : "same as input";
    }
    else
    {
        // bounding box of the entering image rectangle (pixel-edge coordinates) after the transform
        const double xs[4] = { -0.5, entW - 0.5, -0.5, entW - 0.5 };
        const double ys[4] = { -0.5, -0.5, entH - 0.5, entH - 0.5 };
        double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
        for (int k = 0; k < 4; k++)
        {
            double x, y;
            geometry.apply(xs[k], ys[k], x, y);
            minx = std::min(minx, x);
            maxx = std::max(maxx, x);
            miny = std::min(miny, y);
            maxy = std::max(maxy, y);
        }
        wd = std::max(1.0, std::ceil(maxx - minx - 1e-6));
        hd = std::max(1.0, std::ceil(maxy - miny - 1e-6));
        sx = (wd - 1) * 0.5 - (minx + maxx) * 0.5;
        sy = (hd - 1) * 0.5 - (miny + maxy) * 0.5;
        canvasDesc_ = "auto (bounding box, re-centred)";
    }

    if (wd > kMaxDim || hd > kMaxDim)
    {
        char dims[64];
        std::snprintf(dims, sizeof dims, "%.0fx%.0f", wd, hd);
        throw std::runtime_error(std::string("output canvas ") + dims + " exceeds the " + std::to_string(kMaxDim)
                                 + " limit; use a smaller scale");
    }

    const double totW = wd + (usePad_ ? static_cast<double>(padL_) + padR_ : 0.0);
    const double totH = hd + (usePad_ ? static_cast<double>(padT_) + padB_ : 0.0);
    if (totW > kMaxDim || totH > kMaxDim)
    {
        char dims[64];
        std::snprintf(dims, sizeof dims, "%.0fx%.0f", totW, totH);
        throw std::runtime_error(std::string("output canvas with padding ") + dims + " exceeds the " + std::to_string(kMaxDim) + " limit");
    }
    if (totW * totH * 3 > kMaxOutputBytes)
        throw std::runtime_error("output canvas needs more than 512 MB");

    outW_ = static_cast<int>(wd);
    outH_ = static_cast<int>(hd);
    totalW_ = static_cast<int>(totW);
    totalH_ = static_cast<int>(totH);
    forward_ = Mat3::translation(sx, sy) * pre;
}

void Plan::describe(std::ostream &os) const
{
    os << "in " << srcW_ << "x" << srcH_ << " -> out " << totalW_ << "x" << totalH_
       << "  canvas: " << canvasDesc_ << "\n"
       << "M (forward, in -> out) = [" << forward_.at(0, 0) << ' ' << forward_.at(0, 1) << ' ' << forward_.at(0, 2)
       << "; " << forward_.at(1, 0) << ' ' << forward_.at(1, 1) << ' ' << forward_.at(1, 2) << "]" << std::endl;
    if (useCrop_)
        os << "crop: " << cropX_ << "," << cropY_ << " " << cropW_ << "x" << cropH_ << std::endl;
    if (useResize_)
        os << "resize: " << cropW_ << "x" << cropH_ << " -> " << resizeW_ << "x" << resizeH_ << std::endl;
    if (usePad_)
        os << "pad: L" << padL_ << " T" << padT_ << " R" << padR_ << " B" << padB_ << " value "
           << int(padValue_[0]) << "," << int(padValue_[1]) << "," << int(padValue_[2]) << std::endl;
    if (useYuv_)
        os << "colorspace: RGB -> " << Yuv420::name(yuv_) << " -> RGB" << std::endl;
}

Bitmap Plan::run(const Bitmap &src) const
{
    if (cpu_)
    {
        Bitmap img = src;
        if (useYuv_) // same round trip, in software, before anything else
        {
            const std::vector<uint8_t> yuv = Yuv420::encode(src, yuv_);
            img = Yuv420::decode(yuv.data(), src.width(), src.height(), yuv_);
        }
        if (useCrop_)
            img = img.crop(cropX_, cropY_, cropW_, cropH_);
        Bitmap out = runCpu(img);
        if (usePad_)
            out = out.padded(padL_, padT_, padR_, padB_, padValue_);
        return out;
    }
#ifdef AI2D_STANDALONE
    throw std::runtime_error("built with -DAI2D_STANDALONE: only --cpu is available");
#else
    return runAi2d(src);
#endif
}

// CPU reference: inverse warp, same convention as the matrix handed to AI2D
// (forward in -> out, pixel-index coordinates, y down).
Bitmap Plan::runCpu(const Bitmap &src) const
{
    const Mat3 inv = forward_.inverse();
    const double ia = inv.at(0, 0), ib = inv.at(0, 1), itx = inv.at(0, 2);
    const double ic = inv.at(1, 0), id = inv.at(1, 1), ity = inv.at(1, 2);

    const int W = src.width(), H = src.height();
    Bitmap dst(outW_, outH_, static_cast<uint8_t>(kBoundVal));

    for (int y = 0; y < outH_; y++)
        for (int x = 0; x < outW_; x++)
        {
            const double sx = ia * x + ib * y + itx;
            const double sy = ic * x + id * y + ity;
            const size_t o = static_cast<size_t>(y) * outW_ + x;
            if (nearest_)
            {
                const int ix = static_cast<int>(std::floor(sx + 0.5)), iy = static_cast<int>(std::floor(sy + 0.5));
                if (ix < 0 || ix >= W || iy < 0 || iy >= H)
                    continue;
                for (int ch = 0; ch < 3; ch++)
                    dst.plane(ch)[o] = src.plane(ch)[static_cast<size_t>(iy) * W + ix];
            }
            else
            {
                if (sx < -0.5 || sx > W - 0.5 || sy < -0.5 || sy > H - 0.5)
                    continue;
                const double fx = std::min(std::max(sx, 0.0), static_cast<double>(W - 1));
                const double fy = std::min(std::max(sy, 0.0), static_cast<double>(H - 1));
                const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
                const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
                const double wx = fx - x0, wy = fy - y0;
                for (int ch = 0; ch < 3; ch++)
                {
                    const uint8_t *pl = src.plane(ch);
                    const double v = (1 - wy) * ((1 - wx) * pl[static_cast<size_t>(y0) * W + x0] + wx * pl[static_cast<size_t>(y0) * W + x1])
                                   + wy * ((1 - wx) * pl[static_cast<size_t>(y1) * W + x0] + wx * pl[static_cast<size_t>(y1) * W + x1]);
                    dst.plane(ch)[o] = static_cast<uint8_t>(std::lround(v));
                }
            }
        }
    return dst;
}

#ifndef AI2D_STANDALONE
Bitmap Plan::runAi2d(const Bitmap &src) const
{
    using namespace nncase;
    using namespace nncase::runtime;
    using namespace nncase::runtime::k230;
    using namespace nncase::F::k230;

    const int W = src.width(), H = src.height();
    const std::vector<float> M = forward_.params();

    // Input: planar RGB, or the YUV 4:2:0 buffer (stored as one plane of 3H/2 rows).
    std::vector<uint8_t> yuv;
    if (useYuv_)
        yuv = Yuv420::encode(src, yuv_);
    const uint8_t *inData = useYuv_ ? yuv.data() : src.data();
    const size_t inBytes = useYuv_ ? yuv.size() : src.size();

    dims_t in_shape = useYuv_ ? dims_t { 1, 1, (size_t)H * 3 / 2, (size_t)W } : dims_t { 1, 3, (size_t)H, (size_t)W };
    dims_t out_shape { 1, 3, (size_t)totalH_, (size_t)totalW_ }; // canvas plus pad border

    auto in_t = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");
    {
        auto buf = in_t.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
        memcpy(reinterpret_cast<char *>(buf.data()), inData, inBytes);
    }
    hrt::sync(in_t, sync_op_t::sync_write_back, true).expect("write back input failed");

    auto out_t = host_runtime_tensor::create(typecode_t::dt_uint8, out_shape, hrt::pool_shared).expect("cannot create output tensor");
    {
        // pre-fill so pixels the hardware leaves untouched are defined
        auto buf = out_t.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
        memset(reinterpret_cast<char *>(buf.data()), 0, (size_t)3 * totalW_ * totalH_);
    }
    hrt::sync(out_t, sync_op_t::sync_write_back, true).expect("write back output failed");

    const uint32_t cordRound = nearest_ ? 2 : 0;

    // With --colorspace the source is YUV; the destination stays planar RGB, which
    // makes the hardware colour-space converter turn it back into RGB.
    ai2d_format srcFormat = ai2d_format::NCHW_FMT;
    if (useYuv_)
    {
        srcFormat = yuv_ == Yuv420::Format::NV12 ? ai2d_format::YUV420_NV12
            : yuv_ == Yuv420::Format::NV21       ? ai2d_format::YUV420_NV21
                                                 : ai2d_format::YUV420_I420;
    }
    ai2d_datatype_t ai2d_dtype { srcFormat, ai2d_format::NCHW_FMT, typecode_t::dt_uint8, typecode_t::dt_uint8 };
    // crop is applied first; with --crop the matrix is in window coordinates
    ai2d_crop_param_t crop_param { useCrop_, cropX_, cropY_, cropW_, cropH_ };
    ai2d_shift_param_t shift_param { false, 0 };
    // paddings are {N, C, H, W} before/after pairs; the pad goes around the canvas
    ai2d_pad_param_t pad_param { usePad_, { { 0, 0 }, { 0, 0 }, { padT_, padB_ }, { padL_, padR_ } }, ai2d_pad_mode::constant,
        { padValue_[0], padValue_[1], padValue_[2] } };
    // resize_flag=false, but update_static_param derives the interpolation bit from
    // resize_param.interp_method, not affine_param.interp_method, so set both alike.
    const ai2d_interp_method im = nearest_ ? ai2d_interp_method::tf_nearest : ai2d_interp_method::tf_bilinear;
    // AI2D refuses resize and affine in the same pass, so --resize uses the resize
    // unit only when there is no transform; otherwise it is folded into the matrix.
    ai2d_resize_param_t resize_param { mode_ == Mode::Resize, im, ai2d_interp_mode::half_pixel };
    // {flag, interp_method, cord_round, bound_ind, bound_val, bound_smooth, M}
    ai2d_affine_param_t affine_param { mode_ == Mode::Affine,
        nearest_ ? ai2d_interp_method::cv2_nearest : ai2d_interp_method::cv2_bilinear,
        cordRound, static_cast<uint32_t>(kBoundInd), static_cast<int32_t>(kBoundVal), static_cast<uint32_t>(kBoundSmooth), M };

    ai2d_builder builder { in_shape, out_shape, ai2d_dtype, crop_param, shift_param, pad_param, resize_param, affine_param };
    builder.build_schedule().expect("ai2d build_schedule failed");
    builder.invoke(in_t, out_t).expect("error occurred in ai2d running");

    hrt::sync(out_t, sync_op_t::sync_invalidate, true).expect("invalidate output failed");
    auto obuf = out_t.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();

    Bitmap dst(totalW_, totalH_, 0);
    memcpy(dst.data(), reinterpret_cast<const uint8_t *>(obuf.data()), dst.size());
    return dst;
}
#endif

// ===========================================================================
// TestSuite
// ===========================================================================

namespace
{
const char *const kSuiteInput = "ai2d_test.bmp";
const char *const kSuiteHashFile = "ai2d_test.hash";
const char *const kSuiteCpuHashFile = "ai2d_test_cpu.hash"; // --cpu runs: software hashes never mix with the board's
const char *const kSuiteOutputPrefix = "ai2d_test_output";

// Options of each test, nullptr-terminated. Order matters: it fixes the output
// file numbers and the line numbers in ai2d_test.hash, so only append.
// Crop windows are chosen to fit even the smallest accepted image (128x128) and
// to be even-aligned, so the colour-space tests can use them too.
const char *const kInvocations[][16] = {
    // --- rotate ---
    { "--rotate", "45" },                                              // 0
    { "--rotate", "90" },                                              // 1
    { "--rotate", "-30" },                                             // 2
    { "--rotate", "180" },                                             // 3
    { "--rotate", "10", "--interp", "nearest" },                       // 4
    // --- scale ---
    { "--scale", "0.5" },                                              // 5
    { "--scale", "2" },                                                // 6
    { "--scale", "0.75,1.5" },                                         // 7
    { "--scale", "1.5", "--interp", "nearest" },                       // 8
    // --- shear ---
    { "--shear", "0.3,0" },                                            // 9
    { "--shear", "0,0.3" },                                            // 10
    { "--shear", "0.2,0.2" },                                          // 11
    // --- flips ---
    { "--flipx" },                                                     // 12
    { "--flipy" },                                                     // 13
    { "--flipx", "--flipy" },                                          // 14
    // --- translate and raw matrix (output keeps the input size) ---
    { "--translate", "20,-10" },                                       // 15
    { "--translate", "-33,17", "--interp", "nearest" },                // 16
    { "--matrix", "0.9,0.2,10,-0.1,1.1,5" },                           // 17
    { "--matrix", "1,0,0,0,1,0" },                                     // 18 identity
    // --- combined transforms, order sensitive ---
    { "--rotate", "30", "--scale", "1.5", "--shear", "0.2,0" },        // 19
    { "--shear", "0.2,0", "--scale", "1.5", "--rotate", "30" },        // 20
    { "--rotate", "20", "--flipx", "--translate", "12,8" },            // 21
    // --- crop ---
    { "--crop", "16,16,96,96" },                                       // 22
    { "--crop", "0,0,64,64", "--interp", "nearest" },                  // 23
    { "--crop", "32,32,64,64", "--rotate", "45" },                     // 24
    // --- resize ---
    { "--resize", "224,224" },                                         // 25
    { "--resize", "100,60", "--interp", "nearest" },                   // 26
    { "--resize", "320,320" },                                         // 27 upscale
    { "--resize", "64,64" },                                           // 28 strong downscale
    // --- pad ---
    { "--pad", "16" },                                                 // 29
    { "--pad", "4,8,12,16", "--pad-value", "255,0,0" },                // 30
    { "--pad", "10", "--pad-value", "128" },                           // 31
    // --- stages together ---
    { "--crop", "16,16,96,96", "--resize", "224,224", "--pad", "8", "--pad-value", "114" },            // 32
    { "--crop", "16,16,96,96", "--resize", "160,160", "--rotate", "30", "--pad", "8" },                // 33
    { "--resize", "200,200", "--rotate", "45", "--pad", "10", "--pad-value", "0,255,0" },              // 34
    // --- colour-space round trip, each format, alone (copy pass) ---
    { "--colorspace", "NV12" },                                        // 35
    { "--colorspace", "NV21" },                                        // 36
    { "--colorspace", "I420" },                                        // 37
    // --- colour space with transforms and stages ---
    { "--colorspace", "NV12", "--rotate", "45" },                      // 38
    { "--colorspace", "NV21", "--scale", "0.5" },                      // 39
    { "--colorspace", "I420", "--flipx", "--shear", "0.2,0" },         // 40
    { "--colorspace", "NV12", "--crop", "16,16,96,96", "--resize", "224,224" },                        // 41
    { "--colorspace", "I420", "--crop", "32,32,64,64" },               // 42
    { "--colorspace", "NV21", "--resize", "128,128", "--pad", "8" },   // 43
    { "--colorspace", "I420", "--rotate", "30", "--interp", "nearest" }, // 44
    { "--colorspace", "NV12", "--pad", "16" },                         // 45
};
constexpr size_t kInvocationCount = sizeof kInvocations / sizeof kInvocations[0];
constexpr size_t kInvocationMaxArgs = sizeof kInvocations[0] / sizeof kInvocations[0][0];
static_assert(kInvocationMaxArgs >= 14, "room for the longest invocation plus the terminator");
} // namespace

size_t TestSuite::count()
{
    return kInvocationCount;
}

std::string TestSuite::describe(size_t index)
{
    std::string text;
    for (size_t k = 0; k < kInvocationMaxArgs && kInvocations[index][k]; k++)
        text += (k ? " " : "") + std::string(kInvocations[index][k]);
    return text;
}

std::string TestSuite::outputName(size_t index)
{
    return std::string(kSuiteOutputPrefix) + std::to_string(index) + ".bmp";
}

std::string TestSuite::hashFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read back " + path);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const XXH64_hash_t h = XXH3_64bits(bytes.data(), bytes.size());
    char text[17];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(h));
    return text;
}

std::vector<std::string> TestSuite::readHashes(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open " + path + " (create it with --run-test-suite -f)");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

int TestSuite::run(bool writeHashes, bool cpu)
{
    const std::string hashPath = cpu ? kSuiteCpuHashFile : kSuiteHashFile;
    std::vector<std::string> expected;
    Bitmap src;
    try
    {
        if (!writeHashes)
        {
            expected = readHashes(hashPath);
            if (expected.size() != kInvocationCount)
                throw std::runtime_error(hashPath + " has " + std::to_string(expected.size()) + " lines but the suite has "
                                         + std::to_string(kInvocationCount) + " tests (regenerate it with -f)");
        }

        src.load(kSuiteInput);
        if (src.width() < 128 || src.height() < 128 || src.width() > 640 || src.height() > 640)
            throw std::runtime_error(std::string(kSuiteInput) + " is " + std::to_string(src.width()) + "x" + std::to_string(src.height())
                                     + ", the suite needs 128x128 .. 640x640");
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "suite: " << kInvocationCount << " tests on " << kSuiteInput << " (" << src.width() << "x" << src.height() << ")"
              << (cpu ? ", software renderer" : "") << (writeHashes ? ", writing " : ", verifying against ") << hashPath << std::endl;

    std::vector<std::string> hashes;
    size_t passed = 0, failed = 0, errors = 0;

    for (size_t i = 0; i < kInvocationCount; i++)
    {
        const std::string label = "#" + std::to_string(i);
        const std::string args = describe(i) + (cpu ? " --cpu" : "");
        try
        {
            // argv as the normal command line would see it, with the suite's file names
            std::vector<std::string> words { "ai2d_test", kSuiteInput, outputName(i) };
            for (size_t k = 0; k < kInvocationMaxArgs && kInvocations[i][k]; k++)
                words.push_back(kInvocations[i][k]);
            if (cpu)
                words.push_back("--cpu");
            std::vector<char *> argv;
            for (std::string &w : words)
                argv.push_back(&w[0]);

            CommandLine cmd;
            int exitCode = 0;
            if (!CommandLine::parse(static_cast<int>(argv.size()), argv.data(), cmd, exitCode) || cmd.runSuite)
                throw std::runtime_error("invalid suite entry");

            Plan plan;
            plan.build(cmd, src.width(), src.height());
            plan.run(src).save(cmd.output);
            const std::string hash = hashFile(cmd.output);
            hashes.push_back(hash);

            if (writeHashes)
            {
                std::cout << label << "  " << hash << "  SAVED    " << args << std::endl;
            }
            else if (hash == expected[i])
            {
                passed++;
                std::cout << label << "  " << hash << "  SUCCESS  " << args << std::endl;
            }
            else
            {
                failed++;
                std::cout << label << "  " << hash << "  FAIL     " << args << "  (expected " << expected[i] << ")" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            errors++;
            hashes.push_back("");
            std::cout << label << "  " << std::string(16, '-') << "  ERROR    " << args << "  (" << e.what() << ")" << std::endl;
        }
    }

    if (writeHashes)
    {
        if (errors)
        {
            std::cout << errors << " test(s) failed to run, " << hashPath << " not written" << std::endl;
            return 1;
        }
        std::ofstream f(hashPath, std::ios::trunc);
        for (const std::string &h : hashes)
            f << h << '\n';
        if (!f)
        {
            std::cerr << "error: cannot write " << hashPath << std::endl;
            return 1;
        }
        std::cout << "wrote " << hashPath << " (" << hashes.size() << " hashes)" << std::endl;
        return 0;
    }

    std::cout << kInvocationCount << " tests: " << passed << " SUCCESS, " << failed << " FAIL, " << errors << " ERROR" << std::endl;
    return (failed || errors) ? 1 : 0;
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char **argv)
{
    CommandLine cmd;
    int exitCode = 0;
    if (!CommandLine::parse(argc, argv, cmd, exitCode))
        return exitCode;
    if (cmd.runSuite)
        return TestSuite::run(cmd.writeHashes, cmd.cpu);

    try
    {
        Bitmap src;
        src.load(cmd.input);

        Plan plan;
        plan.build(cmd, src.width(), src.height());
        plan.describe(std::cout);
        if (cmd.dryRun)
            return 0;

        plan.run(src).save(cmd.output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
