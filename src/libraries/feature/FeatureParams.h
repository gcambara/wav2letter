/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <math.h>
#include <stdint.h>

namespace w2l {

enum class WindowType {
  HAMMING = 0,
  HANNING = 1,
};
enum class FrequencyScale {
  MEL = 0,
  LINEAR = 1,
  LOG10 = 2,
};

struct FeatureParams {
  // frequency (Hz) of speech signal recording
  int64_t samplingFreq;

  // frame size in milliseconds
  int64_t frameSizeMs;

  // frame step size in milliseconds
  int64_t frameStrideMs;

  // number of filterbank channels
  // Kaldi recommends using 23 for 16KHz and 15 for 8KHz sampled data
  int64_t numFilterbankChans;

  // lower cutoff frequency (HZ) for the filterbank
  int64_t lowFreqFilterbank;

  // upper cutoff frequency (HZ) for the filterbank
  int64_t highFreqFilterbank;

  // number of cepstral coefficients
  int64_t numCepstralCoeffs;

  // liftering parameter
  int64_t lifterParam;

  //  number of delta (first order regression) coefficients
  int64_t deltaWindow;

  //  number of acceleration (second order regression) coefficients
  int64_t accWindow;

  // pitch frame size in milliseconds
  int64_t pitchFrameSizeMs;

  // pitch frame stride in milliseconds
  int64_t pitchFrameStrideMs;

  // low pass filter cutoff in Hertz
  float lowPassCutoff;

  // low pass filter width
  int64_t lowPassFilterWidth;

  // minimum possible frequency value in Hertz
  float minF0;

  // maximum possible frequency value in Hertz
  float maxF0;

  // minimum f0, applied in soft way, must not exceed min-f0
  float softMinF0;

  // increasing this factor reduces NCCF for quiet frames
  float nccfBallast;

  // factor that penalizes frequency change
  float penaltyFactor;

  // smallest relative change in pitch measured by the algorithm
  float deltaPitch;

  // sample frequency for NCCF, must exceed twice lowpass-cutoff
  float resampleFrequency;

  // integer that determines filter width when upsampling NCCF
  int64_t upsampleFilterWidth;

  // time in milliseconds to average voice quality parameters
  int64_t voiceQualityAverageMs;

  // analysis window function handle for framing (hamming by default)
  WindowType windowType;

  // preemphasis filtering coefficient (0.7 default)
  float preemCoef;

  // option controlling the size of the mel floor (1.0 default)
  float melFloor;

  // dithering constant (0.0 default ==> no dithering)
  float ditherVal;

  // use power instead of magnitude for filterbank energies
  bool usePower;

  // If true, append log energy term as a feature to MFSC
  // For MFCC, C0 is replaced with energy term
  bool useEnergy;

  // If true, use energy before PreEmphasis and Windowing
  bool rawEnergy;

  // If true, remove DC offset from the signal frames
  bool zeroMeanFrame;

  FeatureParams(
      int64_t samplingfreq = 16000,
      int64_t framesizems = 25,
      int64_t framestridems = 10,
      int64_t numfilterbankchans = 23,
      int64_t lowfreqfilterbank = 0.0,
      int64_t highfreqfilterbank = -1.0, // If -ve value, then samplingFreq/2
      int64_t numcepstralcoeffs = 13,
      int64_t lifterparam = 22,
      int64_t deltawindow = 2,
      int64_t accwindow = 2,
      int64_t pitchframesizems = 25,
      int64_t pitchframestridems = 10,
      float lowpasscutoff = 1000,
      int64_t lowpassfilterwidth = 2,
      float minf0 = 50,
      float maxf0 = 400,
      float softminf0 = 10,
      float nccfballast = 0.625,
      float penaltyfactor = 0.1,
      float deltapitch = 0.005,
      float resamplefrequency = 4000,
      int64_t upsamplefilterwidth = 5,
      int64_t voicequalityaveragems = 500,
      WindowType windowtype = WindowType::HAMMING,
      float preemcoef = 0.97,
      float melfloor = 1.0,
      float ditherval = 0.0,
      bool usepower = true,
      bool usenergy = true,
      bool rawenergy = true,
      bool zeromeanframe = true)
      : samplingFreq(samplingfreq),
        frameSizeMs(framesizems),
        frameStrideMs(framestridems),
        numFilterbankChans(numfilterbankchans),
        lowFreqFilterbank(lowfreqfilterbank),
        highFreqFilterbank(highfreqfilterbank),
        numCepstralCoeffs(numcepstralcoeffs),
        lifterParam(lifterparam),
        deltaWindow(deltawindow),
        accWindow(accwindow),
        pitchFrameSizeMs(pitchframesizems),
        pitchFrameStrideMs(pitchframestridems),
        lowPassCutoff(lowpasscutoff),
        lowPassFilterWidth(lowpassfilterwidth),
        minF0(minf0),
        maxF0(maxf0),
        softMinF0(softminf0),
        nccfBallast(nccfballast),
        penaltyFactor(penaltyfactor),
        deltaPitch(deltapitch),
        resampleFrequency(resamplefrequency),
        upsampleFilterWidth(upsamplefilterwidth),
        voiceQualityAverageMs(voicequalityaveragems),
        windowType(windowtype),
        preemCoef(preemcoef),
        melFloor(melfloor),
        ditherVal(ditherval),
        usePower(usepower),
        useEnergy(usenergy),
        rawEnergy(rawenergy),
        zeroMeanFrame(zeromeanframe) {}

  // frame size (no of samples)
  // the last frame is discarded, if less than the frame size
  int64_t numFrameSizeSamples() const {
    return static_cast<int>(round(1e-3 * frameSizeMs * samplingFreq));
  }

  int64_t numFrameStrideSamples() const {
    return static_cast<int>(round(1e-3 * frameStrideMs * samplingFreq));
  }

  int64_t nFft() const {
    int64_t nsamples = numFrameSizeSamples();
    return (nsamples > 0) ? 1 << static_cast<int>(ceil(log2(nsamples))) : 0;
  }

  int64_t filterFreqResponseLen() const {
    return (nFft() >> 1) + 1;
  }

  int64_t powSpecFeatSz() const {
    return filterFreqResponseLen();
  }

  int64_t mfscFeatSz() const {
    int64_t devMultiplier =
        1 + (deltaWindow > 0 ? 1 : 0) + (accWindow > 0 ? 1 : 0);
    return (numFilterbankChans + (useEnergy ? 1 : 0)) * (devMultiplier);
  }

  int64_t mfccFeatSz() const {
    int64_t devMultiplier =
        1 + (deltaWindow > 0 ? 1 : 0) + (accWindow > 0 ? 1 : 0);
    return numCepstralCoeffs * devMultiplier;
  }

  int64_t pitchFeatSz() const {
    // Three pitch features: pitch, POV and deltaPitch.
    return 3;
  }

  int64_t numFrames(int64_t inSize) const {
    auto frameSize = numFrameSizeSamples();
    auto frameStride = numFrameStrideSamples();
    if (frameStride <= 0 || inSize < frameSize) {
      return 0;
    }
    return 1 + floor((inSize - frameSize) * 1.0 / frameStride);
  }
};

} // namespace w2l
