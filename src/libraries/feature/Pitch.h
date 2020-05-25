/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <cstddef>
#include <vector>
#include "SpeechUtils.h"
#include "FeatureParams.h"
#include "common/FlashlightUtils.h"
#include "Derivatives.h"

namespace w2l {

// Computes pitch features for a speech signal.

template <typename T>
class Pitch {
 public:
  explicit Pitch(const FeatureParams& params);
  //void apply(const std::vector<T>& input, std::vector<T> &pitch, std::vector<T> &pov, std::vector<T> &deltaPitch);
  std::vector<T> apply(const std::vector<T>& input, std::vector<T> &features, int64_t nFrames, int64_t numFeat);
 private:
  std::vector<T> normalizeRMS(std::vector<T> const &input) const;
  T computeRMS(std::vector<T> const &input) const;
  std::vector<std::vector<T>> computeNCCFs(std::vector<T> const &input, T nccfBallast) const;
  void getPitchVector(std::vector<int> const &optimalPitchTrajectory, std::vector<T> &pitch);
  std::vector<T> getNccfsAtOptimalLags(std::vector<int> const &optimalPitchTrajectory, std::vector<std::vector<T>> const &nccfs) const;
  void getPovFeatures(std::vector<T> const &optimalNccfs, std::vector<T> &pov);
  void getDeltaPitch(std::vector<T> const &pitch, std::vector<T> &deltaPitch);
  void postProcessPitch(std::vector<T> &pitch);

  // Voice quality functions.
  std::vector<T> getJitterAbsolute(std::vector<T> const &pitch) const;
  std::vector<T> getJitterRelative(std::vector<T> const &pitch) const;
  std::vector<T> getShimmerDb(std::vector<T> const &input, int64_t nFrames) const;
  std::vector<T> getShimmerRelative(std::vector<T> const &input, int64_t nFrames) const;

  // Math utility functions.
  std::vector<T> addValueToVector(std::vector<T> const &input, T value) const;
  T gcd(int a, int b) const;
  T lcm(int a, int b) const;
  void addVectorToVector(std::vector<T> &vectorA, std::vector<T> const &vectorB, T factor);
  void addVectorsProductToVector(std::vector<T> &vectorA, std::vector<T> const &vectorB, std::vector<T> const &vectorC, T productFactor, T sumFactor);

  FeatureParams featParams_;

  // NFCC variables.
  T minLag_;
  T maxLag_;
  T upsampleFilterFrequency_;
  T filterWidth_;
  T outerMinLag_;
  T outerMaxLag_;
  T outerMinLagSamples_;
  T outerMaxLagSamples_;
  T windowSizeSamples_;
  T windowShiftSamples_;

  // Viterbi variables.
  std::vector<std::vector<T>> transitionCosts_;

  //Kaldi-like functions.
  int32_t num_samples_in_;
  int64_t input_samples_in_unit_;
  int64_t output_samples_in_unit_;
  int64_t samp_rate_in_;
  int64_t samp_rate_out_;
  int64_t arbitrary_samp_rate_in_;
  int64_t input_sample_offset_;  ///< The number of input samples we have
                               ///< already received for this signal
                               ///< (including anything in remainder_)
  int64_t output_sample_offset_;  ///< The number of samples we have already
                                ///< output for this signal.
  std::vector<T> lags_;
  // the first lag of the downsampled signal at which we measure NCCF
  int32_t nccf_first_lag_;
  // the last lag of the downsampled signal at which we measure NCCF
  int32_t nccf_last_lag_;
  std::vector<T> input_remainder_;  ///< A small trailing part of the
                                       ///< previously seen input signal.
  std::vector<int64_t> first_index_;  // The first input-sample index that we sum
                                    // over, for this output-sample index.
  std::vector<int64_t> first_arbitrary_index_;
  std::vector<std::vector<T>> weights_;
  std::vector<std::vector<T>> arbitraryWeights_;
  std::vector<T> forward_cost_;

  std::vector<T> linearResample(const std::vector<T>& input);
  std::vector<std::vector<T>> arbitraryResample(const std::vector<std::vector<T>>& input);
  std::vector<T> optimizePitchTrajectory(const std::vector<std::vector<T>> &nccfs, const T &framesNumber, const T &lagsNumber);
  std::vector<std::vector<T>> computeLocalCosts(const std::vector<std::vector<T>> &nccfs);
  std::vector<std::vector<T>> computeTransitionCosts();
  std::vector<int> computeOptimalPitchTrajectory(const std::vector<std::vector<T>> &localCosts);

  int64_t debug_counter;

  void setIndexesAndWeights();
  void setArbitraryIndexes(const std::vector<T> &sample_points);
  void setArbitraryWeights(const std::vector<T> &sample_points);
  int32_t numSamplesIn() const { return num_samples_in_; }
  int32_t numSamplesOut() const { return arbitraryWeights_.size(); }
  T lowPassFilter(T, T cutoff, T width) const;

  /// This function outputs the number of output samples we will output
  /// for a signal with "input_num_samp" input samples.  If flush == true,
  /// we return the largest n such that
  /// (n/samp_rate_out_) is in the interval [ 0, input_num_samp/samp_rate_in_ ),
  /// and note that the interval is half-open.  If flush == false,
  /// define window_width as num_zeros / (2.0 * filter_cutoff_);
  /// we return the largest n such that (n/samp_rate_out_) is in the interval
  /// [ 0, input_num_samp/samp_rate_in_ - window_width ).
  int64_t getNumOutputSamples(int64_t input_num_samp) const;

  /// Given an output-sample index, this function outputs to *first_samp_in the
  /// first input-sample index that we have a weight on (may be negative),
  /// and to *samp_out_wrapped the index into weights_ where we can get the
  /// corresponding weights on the input.
  inline void getIndexes(int64_t samp_out,
                         int64_t *first_samp_in,
                         int64_t *samp_out_wrapped) const;

  void setRemainder(const std::vector<T> &input);

  std::vector<T> selectLags();
};
} // namespace w2l