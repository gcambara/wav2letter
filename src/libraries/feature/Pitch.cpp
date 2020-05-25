/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Pitch.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <limits>
#include "SpeechUtils.h"

#include <fstream>
#include <iostream>
#include <iterator>

namespace w2l {

template <typename T>
Pitch<T>::Pitch(const FeatureParams& params)
    : featParams_(params){
    samp_rate_in_ = this->featParams_.samplingFreq;
    samp_rate_out_ = this->featParams_.resampleFrequency;
    arbitrary_samp_rate_in_ = this->featParams_.resampleFrequency;

    int64_t base_freq = gcd(samp_rate_in_, samp_rate_out_);
    input_samples_in_unit_ = samp_rate_in_ / base_freq;
    output_samples_in_unit_ = samp_rate_out_ / base_freq;

    // Set variables for linear resample
    setIndexesAndWeights();
    input_sample_offset_ = 0;
    output_sample_offset_ = 0;
    input_remainder_.resize(0);

    // Variables for NFCC computation and arbitrary resample.
    minLag_ = 1/this->featParams_.maxF0;
    maxLag_ = 1/this->featParams_.minF0;
    upsampleFilterFrequency_ = this->featParams_.resampleFrequency/2.0;
    filterWidth_ = this->featParams_.upsampleFilterWidth/upsampleFilterFrequency_;
    outerMinLag_ = minLag_ - filterWidth_/2.0;
    outerMaxLag_ = maxLag_ + filterWidth_/2.0;
    nccf_first_lag_ = ceil(this->featParams_.resampleFrequency * outerMinLag_);
    nccf_last_lag_ = floor(this->featParams_.resampleFrequency * outerMaxLag_);
    num_samples_in_ = nccf_last_lag_ + 1 - nccf_first_lag_;
    outerMinLagSamples_ = outerMinLag_*this->featParams_.resampleFrequency;
    outerMaxLagSamples_ = outerMaxLag_*this->featParams_.resampleFrequency;
    windowSizeSamples_ = (this->featParams_.pitchFrameSizeMs/1000.0)*this->featParams_.resampleFrequency;
    windowShiftSamples_ = (this->featParams_.pitchFrameStrideMs/1000.0)*this->featParams_.resampleFrequency;

    // Choose the lags at which we resample the NCCF.
    lags_ = selectLags();
    auto lags_offset = addValueToVector(lags_, -nccf_first_lag_/this->featParams_.resampleFrequency);
    setArbitraryIndexes(lags_offset);
    setArbitraryWeights(lags_offset);

    // Viterbi variables.
    transitionCosts_ = computeTransitionCosts();
}

template <typename T>
std::vector<T> Pitch<T>::apply(const std::vector<T>& input, std::vector<T> &features, int64_t nFrames, int64_t numFeat) {
  auto additionalFeatures = 0;

  std::vector<T> shimmerDb(nFrames);
  if (FLAGS_shimmerdb) {
    shimmerDb = getShimmerDb(input, nFrames);
    additionalFeatures += 1;
  }

  std::vector<T> shimmerRelative(nFrames);
  if (FLAGS_shimmerrelative) {
    shimmerRelative = getShimmerRelative(input, nFrames);
    additionalFeatures += 1;
  }

  auto downsampledWave = linearResample(input);

  auto normalizedSignal = normalizeRMS(downsampledWave);

  auto nccfs = computeNCCFs(normalizedSignal, this->featParams_.nccfBallast);

  auto upsampledNccfs = arbitraryResample(nccfs);

  auto localCosts = computeLocalCosts(upsampledNccfs);

  auto optimalPitchTrajectory = computeOptimalPitchTrajectory(localCosts);

  std::vector<T> pitch(nFrames, 0.0);
  getPitchVector(optimalPitchTrajectory, pitch);
  
  std::vector<T> jitterAbsolute(nFrames);
  if (FLAGS_jitterabsolute) {
    jitterAbsolute = getJitterAbsolute(pitch);
    additionalFeatures += 1;
  }

  std::vector<T> jitterRelative(nFrames);
  if (FLAGS_jitterrelative) {
    jitterRelative = getJitterRelative(pitch);
    additionalFeatures += 1;
  }

  postProcessPitch(pitch);
  additionalFeatures += 1;

  auto nccfsWithoutBallast = computeNCCFs(normalizedSignal, 0.0);

  auto upsampledNccfsWithoutBallast = arbitraryResample(nccfsWithoutBallast);

  auto optimalNccfs = getNccfsAtOptimalLags(optimalPitchTrajectory, upsampledNccfsWithoutBallast);

  std::vector<T> pov(nFrames, 0.0);
  getPovFeatures(optimalNccfs, pov);
  additionalFeatures += 1;

  std::vector<T> deltaPitch(nFrames, 0.0);
  getDeltaPitch(pitch, deltaPitch);
  additionalFeatures += 1;

  // Append pitch features to MFSC.
  size_t start = numFeat;
  for (size_t f = 0; f < nFrames; ++f) {
    features.insert(features.begin() + start, pitch[f - 1]);
    features.insert(features.begin() + start + 1, pov[f - 1]);
    features.insert(features.begin() + start + 2, deltaPitch[f - 1]);

    auto voiceQualityIter = 0;
    if (FLAGS_jitterabsolute) {
      voiceQualityIter += 1;
      features.insert(features.begin() + start + 2 + voiceQualityIter, jitterAbsolute[f - 1]);
    }
    if (FLAGS_jitterrelative) {
      voiceQualityIter += 1;
      features.insert(features.begin() + start + 2 + voiceQualityIter, jitterRelative[f - 1]);
    }
    if (FLAGS_shimmerdb) {
      voiceQualityIter += 1;
      features.insert(features.begin() + start + 2 + voiceQualityIter, shimmerDb[f - 1]);
    }
    if (FLAGS_shimmerrelative) {
      voiceQualityIter += 1;
      features.insert(features.begin() + start + 2 + voiceQualityIter, shimmerRelative[f - 1]);
    }
    if (FLAGS_shimmerapq3) {
      //TO BE IMPLEMENTED
      std::vector<T> shimmerApq3(nFrames);
      voiceQualityIter += 1;
      features.insert(features.begin() + start + 2 + voiceQualityIter, shimmerApq3[f - 1]);
    }

    start += numFeat + additionalFeatures;
  }

  return features;
}

template <typename T>
void Pitch<T>::setIndexesAndWeights() {
  first_index_.resize(2);
  weights_.resize(output_samples_in_unit_);

  double window_width = this->featParams_.lowPassFilterWidth / (2.0 * this->featParams_.lowPassCutoff);

  for (int64_t i = 0; i < output_samples_in_unit_; i++) {
    double output_t = i / static_cast<double>(samp_rate_out_);
    double min_t = output_t - window_width, max_t = output_t + window_width;
    // we do ceil on the min and floor on the max, because if we did it
    // the other way around we would unnecessarily include indexes just
    // outside the window, with zero coefficients.  It's possible
    // if the arguments to the ceil and floor expressions are integers
    // (e.g. if this->featParams_.lowPassCutoff has an exact ratio with the sample rates),
    // that we unnecessarily include something with a zero coefficient,
    // but this is only a slight efficiency issue.
    int64_t min_input_index = ceil(min_t * samp_rate_in_),
        max_input_index = floor(max_t * samp_rate_in_),
        num_indices = max_input_index - min_input_index + 1;
    first_index_[i] = min_input_index;
    weights_[i].resize(num_indices);
    for (int64_t j = 0; j < num_indices; j++) {
      int64_t input_index = min_input_index + j;
      double input_t = input_index / static_cast<double>(samp_rate_in_),
          delta_t = input_t - output_t;
      // sign of delta_t doesn't matter.
      weights_[i][j] = lowPassFilter(delta_t, this->featParams_.lowPassCutoff, this->featParams_.lowPassFilterWidth) / samp_rate_in_;
    }
  }
}

template <typename T>
void Pitch<T>::setArbitraryIndexes(const std::vector<T> &sample_points) {
  int32_t num_samples = sample_points.size();
  first_arbitrary_index_.resize(num_samples);
  arbitraryWeights_.resize(num_samples);
  auto filter_width = this->featParams_.upsampleFilterWidth / (2.0 * upsampleFilterFrequency_);

  for (int32_t  i = 0; i < num_samples; i++) {
    // the t values are in seconds.
    auto t = sample_points[i], t_min = t - filter_width, t_max = t + filter_width;
    int32_t index_min = ceil(arbitrary_samp_rate_in_ * t_min), index_max = floor(arbitrary_samp_rate_in_ * t_max);
    // the ceil on index min and the floor on index_max are because there
    // is no point using indices just outside the window (coeffs would be zero).
    if (index_min < 0)
      index_min = 0;

    if (index_max >= num_samples_in_)
      index_max = num_samples_in_ - 1;

    first_arbitrary_index_[i] = index_min;
    arbitraryWeights_[i].resize(index_max - index_min + 1);
  }
}

template <typename T>
void Pitch<T>::setArbitraryWeights(const std::vector<T> &sample_points) {
  int32_t num_samples_out = numSamplesOut();
  for (int32_t i = 0; i < num_samples_out; i++) {
    for (int32_t j = 0 ; j < arbitraryWeights_[i].size(); j++) {
      auto delta_t = sample_points[i] - ((double)(first_arbitrary_index_[i] + j)/(double)arbitrary_samp_rate_in_);
      // Include at this point the factor of 1.0 / samp_rate_in_ which
      // appears in the math.
      arbitraryWeights_[i][j] = lowPassFilter(delta_t, upsampleFilterFrequency_, this->featParams_.upsampleFilterWidth) / (double)arbitrary_samp_rate_in_;
    }
  }
}

/** Here, t is a time in seconds representing an offset from
    the center of the windowed filter function, and lowPassFilter(t)
    returns the windowed filter function, described
    in the header as h(t) = f(t)g(t), evaluated at t.
*/
template <typename T>
T Pitch<T>::lowPassFilter(T t, T cutoff, T width) const {
  T window(0.0); // raised-cosine (Hanning) window of width
  T filter(2 * cutoff); // width/2*cutoff

  if (fabs(t) < width / (2.0 * cutoff))
    window = 0.5 * (1 + cos(2 * M_PI * cutoff / width * t));

  if (t != 0)
    filter = sin(2 * M_PI * cutoff * t) / (M_PI * t);

  return filter * window;
}


template <typename T>
void Pitch<T>::getPitchVector(std::vector<int> const &optimalPitchTrajectory, std::vector<T> &pitch) {
  for (size_t i = 0; i < pitch.size(); i++) {
    if (i < optimalPitchTrajectory.size())
      pitch[i] = 1.0/lags_[optimalPitchTrajectory[i]];
    else
      //Pad pitch last values with the last obtained pitch value.
      pitch[i] = pitch[optimalPitchTrajectory.size() - 1];
  }
}

template <typename T>
std::vector<T> Pitch<T>::getNccfsAtOptimalLags(std::vector<int> const &optimalPitchTrajectory, std::vector<std::vector<T>> const &nccfs) const {
  std::vector<T> optimalNccfs(optimalPitchTrajectory.size());

  for (size_t i = 0; i < optimalPitchTrajectory.size(); i++) {
    optimalNccfs[i] = nccfs[i][optimalPitchTrajectory[i]];
  }

  return optimalNccfs;
}

template <typename T>
void Pitch<T>::getPovFeatures(std::vector<T> const &optimalNccfs, std::vector<T> &pov) {
  for (size_t i = 0; i < optimalNccfs.size(); i++) {
    auto c = optimalNccfs[i];
    if (c > 1.0)
      c = 1.0;
    else if (c < -1.0)
      c = -1.0;
    pov[i] = 2.0*(pow(1.0001 - c, 0.15) - 1.0);
  } 
}

template <typename T>
void Pitch<T>::getDeltaPitch(std::vector<T> const &pitch, std::vector<T> &deltaPitch) {
  for (size_t i = 1; i < pitch.size() - 1; i++) {
    deltaPitch[i] = (pitch[i + 1] - pitch[i - 1])/2.0;
  }
  
  // Pad the first value with the second one, and the last with its previous one.
  deltaPitch[0] = deltaPitch[1];
  deltaPitch[deltaPitch.size() - 1] = deltaPitch[deltaPitch.size() - 2];
}

template <typename T>
void Pitch<T>::postProcessPitch(std::vector<T> &pitch) {
  for (size_t i = 0; i < pitch.size(); i++) {
    pitch[i] = std::log(pitch[i]);
  }
}

template <typename T>
std::vector<T> Pitch<T>::getJitterAbsolute(std::vector<T> const &pitch) const {
  std::vector<T> jitterAbsolute(pitch.size(), 0.0);
  std::vector<T> pitchPeriods(pitch.size(), 0.0);

  // First of all, compute the pitch periods, computing the inverse of the pitch
  // fundamental frequencies.
  for (size_t i = 0; i < pitchPeriods.size(); i++) {
    pitchPeriods[i] = 1.0/pitch[i];
  }

  // Compute jitter values averaged by the voice quality average time in milliseconds.
  size_t extractedPitchPeriodsNumber = (size_t)ceil((this->featParams_.voiceQualityAverageMs - this->featParams_.frameSizeMs)/(double)this->featParams_.frameStrideMs);

  if (jitterAbsolute.size() < extractedPitchPeriodsNumber) {
    extractedPitchPeriodsNumber = jitterAbsolute.size();
  }

  for (size_t i = 0; i < pitchPeriods.size() - 2; i++) {
    T sum = 0.0;
    T n = 0.0;
    for (size_t j = i; j < extractedPitchPeriodsNumber - 1 + i; ++j) {
      T firstPitchPeriod = pitchPeriods[j];
      T secondPitchPeriod = pitchPeriods[j + 1];
      sum += fabs(firstPitchPeriod - secondPitchPeriod);

      n += 1.0;
    }

    if (n > 1.0) // security
      jitterAbsolute[i] = sum/(n - 1.0);

    if (std::isnan(jitterAbsolute[i])) {
      std::cout << "jitterAbsolute[i]: " << jitterAbsolute[i] << "\n";
      std::cout << "sum: " << sum << "\n";
      std::cout << "n: " << n << "\n";
      std::cout << "jitterAbsolute: " << jitterAbsolute.size() << "\n";
      for (size_t i = 0; i < jitterAbsolute.size(); i++)
        std::cout << "  " << jitterAbsolute[i];
      std::cout << "FINISH " << "\n";

      std::cout << "pitchPeriods: " << pitchPeriods.size() << "\n";
      for (size_t i = 0; i < pitchPeriods.size(); i++)
        std::cout << "  " << pitchPeriods[i];
      std::cout << "FINISH " << "\n";
    }

    if (extractedPitchPeriodsNumber + i - 1 == pitchPeriods.size() - 1)
      extractedPitchPeriodsNumber -= 1;
  }

  // Pad the last two values with the value of the third last one.
  if (jitterAbsolute.size() >= 3) {
    jitterAbsolute[jitterAbsolute.size() - 1] = jitterAbsolute[jitterAbsolute.size() - 3];
    jitterAbsolute[jitterAbsolute.size() - 2] = jitterAbsolute[jitterAbsolute.size() - 3];
  }

  return jitterAbsolute;
}

template <typename T>
std::vector<T> Pitch<T>::getJitterRelative(std::vector<T> const &pitch) const {
  std::vector<T> jitterRelative(pitch.size(), 0.0);
  std::vector<T> pitchPeriods(pitch.size(), 0.0);

  // First of all, compute the pitch periods, computing the inverse of the pitch
  // fundamental frequencies.
  for (size_t i = 0; i < pitchPeriods.size(); i++) {
    pitchPeriods[i] = 1.0/pitch[i];
  }

  // Compute jitter values averaged by the voice quality average time in milliseconds.
  size_t extractedPitchPeriodsNumber = (size_t)ceil((this->featParams_.voiceQualityAverageMs - this->featParams_.frameSizeMs)/(double)this->featParams_.frameStrideMs);

  if (jitterRelative.size() < extractedPitchPeriodsNumber) {
    extractedPitchPeriodsNumber = jitterRelative.size();
  }

  for (size_t i = 0; i < pitchPeriods.size() - 2; i++) {
    T sumNumerator = 0.0;
    T sumDenominator = 0.0;
    T n = 0.0;
    for (size_t j = i; j < extractedPitchPeriodsNumber - 1 + i; ++j) {
      sumNumerator += fabs(pitchPeriods[j] - pitchPeriods[j + 1]);
      sumDenominator += pitchPeriods[j];

      if (j + 1 >= extractedPitchPeriodsNumber - 1 + i)
        sumDenominator += pitchPeriods[j + 1];

      n += 1.0;
    }

    if (n > 1.0) // security
      jitterRelative[i] = ((sumNumerator/(n - 1.0))/(sumDenominator/n))*100.0; // expressed as a percentage

    if (std::isnan(jitterRelative[i])) {
      std::cout << "jitterRelative[i]: " << jitterRelative[i] << "\n";
      std::cout << "sum: " << sumNumerator << "\n";
      std::cout << "n: " << n << "\n";
      std::cout << "jitterRelative: " << jitterRelative.size() << "\n";
      for (size_t i = 0; i < jitterRelative.size(); i++)
        std::cout << "  " << jitterRelative[i];
      std::cout << "FINISH " << "\n";

      std::cout << "pitchPeriods: " << pitchPeriods.size() << "\n";
      for (size_t i = 0; i < pitchPeriods.size(); i++)
        std::cout << "  " << pitchPeriods[i];
      std::cout << "FINISH " << "\n";
    }

    if (extractedPitchPeriodsNumber + i - 1 == pitchPeriods.size() - 1)
      extractedPitchPeriodsNumber -= 1;
  }

  // Pad the last two values with the value of the third last one.
  if (jitterRelative.size() >= 3) {
    jitterRelative[jitterRelative.size() - 1] = jitterRelative[jitterRelative.size() - 3];
    jitterRelative[jitterRelative.size() - 2] = jitterRelative[jitterRelative.size() - 3];
  }

  return jitterRelative;
}

template <typename T>
std::vector<T> Pitch<T>::getShimmerDb(std::vector<T> const &input, int64_t nFrames) const {
  std::vector<T> shimmerDb(nFrames, 0.0);

  // First, frame the signal, keeping record of the peak to peak amplitude.
  std::vector<T> peakToPeakAmplitude(nFrames, 0.0);
  for (size_t i = 0; i < nFrames; i++) {
    typename std::vector<T>::const_iterator first = input.begin() + i*this->featParams_.frameStrideMs*0.001*samp_rate_in_;
    typename std::vector<T>::const_iterator last = input.begin() + i*this->featParams_.frameStrideMs*0.001*samp_rate_in_ + this->featParams_.frameSizeMs*0.001*samp_rate_in_;
    std::vector<T> frame(first, last);

    peakToPeakAmplitude[i] = *std::max_element(first, last) - *std::min_element(first, last);
  }

  size_t extractedPitchPeriodsNumber = (size_t)ceil((this->featParams_.voiceQualityAverageMs - this->featParams_.frameSizeMs)/(double)this->featParams_.frameStrideMs);

  if (shimmerDb.size() < extractedPitchPeriodsNumber) {
    extractedPitchPeriodsNumber = shimmerDb.size();
  }

  for (size_t i = 0; i < nFrames - 2; i++) {
    T sum = 0.0;
    T n = 0.0;
    for (size_t j = i; j < extractedPitchPeriodsNumber - 1 + i; ++j) {
      T firstAmplitude = peakToPeakAmplitude[j];
      T secondAmplitude = peakToPeakAmplitude[j + 1];
      
      if ((firstAmplitude != 0.0) && (secondAmplitude != 0.0))
        sum += fabs(20.0*log(secondAmplitude/firstAmplitude));

      n += 1.0;
    }

    shimmerDb[i] = sum/(n - 1.0);

    if (extractedPitchPeriodsNumber + i - 1 == nFrames - 1)
      extractedPitchPeriodsNumber -= 1;
  }

  // Pad last two values with the value in the last third one.
  shimmerDb[shimmerDb.size() - 1] = shimmerDb[shimmerDb.size() - 3];
  shimmerDb[shimmerDb.size() - 2] = shimmerDb[shimmerDb.size() - 3];

  return shimmerDb;
}

template <typename T>
std::vector<T> Pitch<T>::getShimmerRelative(std::vector<T> const &input, int64_t nFrames) const {
  std::vector<T> shimmerRelative(nFrames, 0.0);

  // First, frame the signal, keeping record of the peak to peak amplitude.
  std::vector<T> peakToPeakAmplitude(nFrames, 0.0);
  for (size_t i = 0; i < nFrames; i++) {
    typename std::vector<T>::const_iterator first = input.begin() + i*this->featParams_.frameStrideMs*0.001*samp_rate_in_;
    typename std::vector<T>::const_iterator last = input.begin() + i*this->featParams_.frameStrideMs*0.001*samp_rate_in_ + this->featParams_.frameSizeMs*0.001*samp_rate_in_;
    std::vector<T> frame(first, last);

    peakToPeakAmplitude[i] = *std::max_element(first, last) - *std::min_element(first, last);

  }

  size_t extractedPitchPeriodsNumber = (size_t)ceil((this->featParams_.voiceQualityAverageMs - this->featParams_.frameSizeMs)/(double)this->featParams_.frameStrideMs);

  if (shimmerRelative.size() < extractedPitchPeriodsNumber) {
    extractedPitchPeriodsNumber = shimmerRelative.size();
  }

  for (size_t i = 0; i < nFrames - 2; i++) {
    T sumNumerator = 0.0;
    T sumDenominator = 0.0;
    T n = 0.0;
    for (size_t j = i; j < extractedPitchPeriodsNumber - 1 + i; ++j) {
      T firstAmplitude = peakToPeakAmplitude[j];
      T secondAmplitude = peakToPeakAmplitude[j + 1];
      
      if ((firstAmplitude != 0.0) && (secondAmplitude != 0.0)) {
        sumNumerator += fabs(firstAmplitude - secondAmplitude);
        sumDenominator += firstAmplitude;

        // For last iteration, sum the second amplitude as well.
        if (j + 1 >= extractedPitchPeriodsNumber - 1 + i)
          sumDenominator += secondAmplitude;
      }
      n += 1.0;
    }

    if (n > 1.0 && sumDenominator != 0.0) // security
      shimmerRelative[i] = (sumNumerator/(n - 1.0))/(sumDenominator/n)*100.0; //expressed as a percentage

    if (extractedPitchPeriodsNumber + i - 1 == nFrames - 1)
      extractedPitchPeriodsNumber -= 1;
  }

  // Pad last two values with the value in the last third one.
  shimmerRelative[shimmerRelative.size() - 1] = shimmerRelative[shimmerRelative.size() - 3];
  shimmerRelative[shimmerRelative.size() - 2] = shimmerRelative[shimmerRelative.size() - 3];

  return shimmerRelative;
}

template <typename T>
std::vector<std::vector<T>> Pitch<T>::computeLocalCosts(const std::vector<std::vector<T>> &nccfs) {
  std::vector<std::vector<T>> localCosts(nccfs.size(), std::vector<T>(lags_.size()));

  for (size_t i = 0; i < nccfs.size(); i++) {
    std::vector<T> localCost(nccfs[i].size(), 1.0);
    addVectorToVector(localCost, nccfs[i], -1.0);
    addVectorsProductToVector(localCost, nccfs[i], lags_, this->featParams_.softMinF0, 1.0);
    localCosts[i] = localCost;
  }

  return localCosts;
}

template <typename T>
std::vector<std::vector<T>> Pitch<T>::computeTransitionCosts() {
  std::vector<std::vector<T>> transitionCosts(lags_.size(), std::vector<T>(lags_.size()));

  for (size_t i = 0; i < lags_.size(); i++) {
    for (size_t j = i; j < lags_.size(); j++) {
      if (i == j) {
        transitionCosts[i][j] = 0.0;
      }
      else {
        transitionCosts[i][j] = FLAGS_penaltyFactor*pow(log(lags_[i]/lags_[j]), 2.0);
        transitionCosts[j][i] = transitionCosts[i][j];
      }
    }
  }

  return transitionCosts;
}

template <typename T>
std::vector<int> Pitch<T>::computeOptimalPitchTrajectory(const std::vector<std::vector<T>> &localCosts) {
  // Obtain the optimal pitch trajectory applying Viterbi algorithm.
    // Initialize variables for Viterbi optimal path computation.
  auto statesNumber = transitionCosts_.size();
  auto observationsNumber = localCosts.size();
  std::vector<int> viterbiPath(observationsNumber, 0);

  // State Space vector, every state is a lag index.
  std::vector<int> stateSpace(statesNumber);
  std::iota(std::begin(stateSpace), std::end(stateSpace), 0);
  // Initialize t1 and t2 to keep the paths.
  std::vector<std::vector<T>> t1(observationsNumber, std::vector<T>(statesNumber, std::numeric_limits<float>::max()));
  std::vector<std::vector<T>> t2(observationsNumber, std::vector<T>(statesNumber, std::numeric_limits<float>::max()));

  for (size_t stateIter = 0; stateIter < statesNumber; ++stateIter) {
    t1[0][stateIter] = localCosts[0][stateIter];
    t2[0][stateIter] = 0;
  }

  for (size_t t = 0; t < observationsNumber - 1; ++t) {
    for (size_t j = 0; j < statesNumber; ++j) {
      for (size_t i = 0; i < statesNumber; ++i) {
        // Store the minimum cost value and previous index.
        if (t1[t + 1][j] > (t1[t][i] + transitionCosts_[j][i])) { 
          t1[t+1][j] = t1[t][i] + transitionCosts_[j][i];
          t2[t+1][j] = i;
        }
      }
      t1[t+1][j] += localCosts[t+1][j];
    }
  }

  auto minValue = t1[t1.size() - 1][0];
  for (size_t i = 1; i < statesNumber - 1; i++) {
    if (t1[t1.size() - 1][i] < minValue) {
      minValue = t1[t1.size() - 1][i];
      viterbiPath[viterbiPath.size() - 1] = i;
    }
  }

  for (size_t j = observationsNumber - 1; j > 0; --j ) {
    viterbiPath[j - 1] = t2[j][viterbiPath[j]];
  }

  return viterbiPath;
}

template <typename T>
std::vector<std::vector<T>> Pitch<T>::arbitraryResample(
    const std::vector<std::vector<T>>& input) {
    std::vector<std::vector<T>> output(input.size(), std::vector<T>(lags_.size()));
    for (size_t i = 0; i < input.size(); i++) {
      for (size_t j = 0; j < lags_.size(); j++) {
        typename std::vector<T>::const_iterator first = input[i].begin() + first_arbitrary_index_[j];
        typename std::vector<T>::const_iterator last = input[i].begin() + first_arbitrary_index_[j] + (int64_t)arbitraryWeights_[j].size();
        std::vector<T> input_part(first, last);
        output[i][j] = std::inner_product(input_part.begin(), input_part.end(), arbitraryWeights_[j].begin(), static_cast<T>(0.0));
      }
    }
    return output;
}

template <typename T>
std::vector<T> Pitch<T>::linearResample(
    const std::vector<T>& input) {
  std::vector<T> output;

  int64_t input_dim = input.size();
  int64_t tot_input_samp = input_sample_offset_ + input_dim,
      tot_output_samp = getNumOutputSamples(tot_input_samp);

  output.resize(tot_output_samp - output_sample_offset_);

  // samp_out is the index into the total output signal, not just the part
  // of it we are producing here.
  for (int64_t samp_out = output_sample_offset_;
       samp_out < tot_output_samp;
       samp_out++) {
    int64_t first_samp_in;
    int64_t samp_out_wrapped;
    getIndexes(samp_out, &first_samp_in, &samp_out_wrapped);
    //const std::vector<T> &weights = weights_[samp_out_wrapped];
    std::vector<T> weights = weights_[samp_out_wrapped];
    // first_input_index is the first index into "input" that we have a weight
    // for.
    int64_t first_input_index = static_cast<int64_t>(first_samp_in -
                                                 input_sample_offset_);
    T this_output;
    if (first_input_index >= 0 &&
        first_input_index + weights.size() <= input_dim) {
      typename std::vector<T>::const_iterator first = input.begin() + first_input_index;
      typename std::vector<T>::const_iterator last = input.begin() + first_input_index + (int64_t)weights.size();
      std::vector<T> input_part(first, last);

      this_output = std::inner_product(input_part.begin(), input_part.end(), weights.begin(), static_cast<T>(0.0));
    } else {  // Handle edge cases.
      this_output = 0.0;
      for (size_t i = 0; i < weights.size(); i++) {
        T weight = weights[i];
        int32_t input_index = first_input_index + i;
        if (input_index < 0 && (int32_t)input_remainder_.size() + input_index >= 0) {
          this_output += weight * input_remainder_[(int32_t)input_remainder_.size() + input_index];
        } else if (input_index >= 0 && input_index < input_dim) {
          this_output += weight * input[input_index];
        }
      }
    }
    int64_t output_index = static_cast<int64_t>(samp_out - output_sample_offset_);
    output[output_index] = this_output;
  }

  setRemainder(input);
  input_sample_offset_ = tot_input_samp;
  output_sample_offset_ = tot_output_samp;

  return output;
}


template <typename T>
void Pitch<T>::setRemainder(const std::vector<T> &input) {
  std::vector<T> old_remainder(input_remainder_);
  // max_remainder_needed is the width of the filter from side to side,
  // measured in input samples.  you might think it should be half that,
  // but you have to consider that you might be wanting to output samples
  // that are "in the past" relative to the beginning of the latest
  // input... anyway, storing more remainder than needed is not harmful.
  int64_t max_remainder_needed = ceil(samp_rate_in_ * this->featParams_.lowPassFilterWidth /
                                    this->featParams_.lowPassCutoff);
  //input_remainder_.Resize(max_remainder_needed);
  input_remainder_.resize(max_remainder_needed);
  for (int64_t index = - input_remainder_.size(); index < 0; index++) {
    // we interpret "index" as an offset from the end of "input" and
    // from the end of input_remainder_.
    int64_t input_index = index + (int64_t)input.size();
    if (input_index >= 0)
      input_remainder_[index + (int64_t)input_remainder_.size()] = input[input_index];
    else if (input_index + (int64_t)old_remainder.size() >= 0)
      input_remainder_[index + (int64_t)input_remainder_.size()] = old_remainder[input_index + (int64_t)old_remainder.size()];
    // else leave it at zero.
  }
}

// inline
template <typename T>
void Pitch<T>::getIndexes(int64_t samp_out,
                                int64_t *first_samp_in,
                                int64_t *samp_out_wrapped) const {
  // A unit is the smallest nonzero amount of time that is an exact
  // multiple of the input and output sample periods.  The unit index
  // is the answer to "which numbered unit we are in".
  int64_t unit_index = samp_out / output_samples_in_unit_;
  // samp_out_wrapped is equal to samp_out % output_samples_in_unit_
  *samp_out_wrapped = static_cast<int64_t>(samp_out -
                                         unit_index * output_samples_in_unit_);
  *first_samp_in = first_index_[*samp_out_wrapped] +
      unit_index * input_samples_in_unit_;
}

template <typename T>
int64_t Pitch<T>::getNumOutputSamples(int64_t input_num_samp) const {
  // For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
  // where tick_freq is the least common multiple of samp_rate_in_ and
  // samp_rate_out_.
  int64_t tick_freq = lcm(samp_rate_in_, samp_rate_out_);
  int64_t ticks_per_input_period = tick_freq / samp_rate_in_;

  // work out the number of ticks in the time interval
  // [ 0, input_num_samp/samp_rate_in_ ).
  int64_t interval_length_in_ticks = input_num_samp * ticks_per_input_period;

  T window_width = this->featParams_.lowPassFilterWidth / (2.0 * this->featParams_.lowPassCutoff);
  // To count the window-width in ticks we take the floor.  This
  // is because since we're looking for the largest integer num-out-samp
  // that fits in the interval, which is open on the right, a reduction
  // in interval length of less than a tick will never make a difference.
  // For example, the largest integer in the interval [ 0, 2 ) and the
  // largest integer in the interval [ 0, 2 - 0.9 ) are the same (both one).
  // So when we're subtracting the window-width we can ignore the fractional
  // part.
  int64_t window_width_ticks = floor(window_width * tick_freq);
  // The time-period of the output that we can sample gets reduced
  // by the window-width (which is actually the distance from the
  // center to the edge of the windowing function) if we're not
  // "flushing the output".
  interval_length_in_ticks -= window_width_ticks;

  if (interval_length_in_ticks <= 0)
    return 0;
  int64_t ticks_per_output_period = tick_freq / samp_rate_out_;
  // Get the last output-sample in the closed interval, i.e. replacing [ ) with
  // [ ].  Note: integer division rounds down.  See
  // http://en.wikipedia.org/wiki/Interval_(mathematics) for an explanation of
  // the notation.
  int64_t last_output_samp = interval_length_in_ticks / ticks_per_output_period;
  // We need the last output-sample in the open interval, so if it takes us to
  // the end of the interval exactly, subtract one.
  if (last_output_samp * ticks_per_output_period == interval_length_in_ticks)
    last_output_samp--;
  // First output-sample index is zero, so the number of output samples
  // is the last output-sample plus one.
  int64_t num_output_samp = last_output_samp + 1;
  return num_output_samp;
}

template <typename T>
std::vector<T> Pitch<T>::normalizeRMS(std::vector<T> const &input) const {
  auto rms = computeRMS(input);

  std::vector<T> normalizedSignal(input.size());
  for (size_t i = 0; i < input.size(); i++) {
      normalizedSignal[i] = input[i]/rms;
  }

  return normalizedSignal;
}

template <typename T>
T Pitch<T>::computeRMS(std::vector<T> const &input) const {
  auto square_sum = 0.0;

  for (size_t i = 0; i < input.size(); i++) {
      square_sum += input[i]*input[i];
  }

  return sqrt(square_sum/input.size());
}

template <typename T>
std::vector<std::vector<T>> Pitch<T>::computeNCCFs(std::vector<T> const &input, T nccfBallast) const {
  auto frameNumber = ceil((input.size() - outerMaxLagSamples_ - windowSizeSamples_)/windowShiftSamples_);
  auto ballastTerm = pow(windowSizeSamples_, 4.0)*nccfBallast;

  // Compute w vectors.
  std::vector<std::vector<T>> nccfs(frameNumber);

  for (size_t t = 0; t < frameNumber; t++) {
    typename std::vector<T>::const_iterator first = input.begin() + t*windowShiftSamples_;
    typename std::vector<T>::const_iterator last = input.begin() + (t*windowShiftSamples_ + windowSizeSamples_ + outerMaxLagSamples_);
    std::vector<T> w(first, last);

    // Compute mean of the w vector, and subtract it to itself.
    auto mean = std::accumulate(std::begin(w), std::end(w), 0.0) / w.size();
    for (size_t i = 0; i < w.size(); i++) {
      w[i] = w[i] - mean;
    } 

    // v0 vector is constant through all lags, do the computations beforehand and save time.
    typename std::vector<T>::const_iterator v0_first = w.begin();
    typename std::vector<T>::const_iterator v0_last = w.begin() + windowSizeSamples_;
    std::vector<T> v0(v0_first, v0_last);

    auto v0Normalization = std::inner_product(v0.begin(), v0.end(), v0.begin(), static_cast<T>(0.0));

    std::vector<T> nccfTime(num_samples_in_, 0.0);
    auto nccfTimeIter = 0;
    for (size_t l = outerMinLagSamples_; l < outerMaxLagSamples_ + 1; l++) {
      typename std::vector<T>::const_iterator vl_first = w.begin() + l;
      typename std::vector<T>::const_iterator vl_last = w.begin() + l + windowSizeSamples_;
      std::vector<T> vl(vl_first, vl_last);

      auto denominator = sqrt((v0Normalization)*(std::inner_product(vl.begin(), vl.end(), vl.begin(), static_cast<T>(0.0))) + ballastTerm);

      if ((double)denominator == 0.0)
        nccfTime[nccfTimeIter] = 0.0;
      else
        nccfTime[nccfTimeIter] = std::inner_product(v0.begin(), v0.end(), vl.begin(), static_cast<T>(0.0))/denominator;

      nccfTimeIter += 1;
    }

    nccfs[t] = nccfTime;
  }

  return nccfs;
}


template <typename T>
std::vector<T> Pitch<T>::selectLags() {
  std::vector<T> lags;
  for (auto lag = minLag_; lag <= maxLag_; lag *= 1.0 + this->featParams_.deltaPitch)
    lags.push_back(lag);

  return lags;
}

template <typename T>
std::vector<T> Pitch<T>::addValueToVector(std::vector<T> const &input, T value) const {
  std::vector<T> newVector(input);
  for (size_t i = 0; i < input.size(); i++) {
    newVector[i] += value; 
  }

  return newVector;
}

/**
   vectorA = vectorA + factor*vectorB
 */
template <typename T>
void Pitch<T>::addVectorToVector(std::vector<T> &vectorA, std::vector<T> const &vectorB, T factor) {
  for (size_t i = 0; i < vectorA.size(); i++) {
    vectorA[i] += factor*vectorB[i]; 
  }
}

/**
   vectorA = sumFactor * vectorA + productFactor * vectorB .* vectorC
 */
template <typename T>
void Pitch<T>::addVectorsProductToVector(std::vector<T> &vectorA, std::vector<T> const &vectorB, std::vector<T> const &vectorC, T productFactor, T sumFactor) {
  for (size_t i = 0; i < vectorA.size(); i++) {
    vectorA[i] = sumFactor*vectorA[i] + productFactor*(vectorB[i]*vectorC[i]);
  }
}

template <typename T>
T Pitch<T>::gcd(int a, int b) const {
   if (b == 0)
   return a;
   return gcd(b, a % b);
}

template <typename T>
T Pitch<T>::lcm(int a, int b) const {
    int temp = gcd(a, b);
    return temp ? (a / temp * b) : 0;
}

template class Pitch<float>;
template class Pitch<double>;
} // namespace w2l
