"""(12 Nov) The original Qwen's profile on the FULL TEST (2,759 x 8), per part: solve-rate bins, thinking, cut-offs.
A .py module because the Colab stage copies only *.py and plan files (test_difficulty.json kept for reading)."""
TEST_DIFFICULTY = {
 "mbpp": {
  "questions": 378,
  "solve": 0.5244708994708994,
  "bins": {
   "0": 0.35714285714285715,
   "1": 0.07407407407407407,
   "2": 0.15873015873015872,
   "3": 0.11904761904761904,
   "4": 0.291005291005291
  },
  "median_think": 1005.0,
  "p90_think": 6806,
  "cutoff": 0.07804232804232804
 },
 "bfcl_simple_python": {
  "questions": 400,
  "solve": 0.8215625,
  "bins": {
   "0": 0.105,
   "1": 0.05,
   "2": 0.0875,
   "3": 0.0925,
   "4": 0.665
  },
  "median_think": 141.0,
  "p90_think": 249,
  "cutoff": 0.0
 },
 "svamp": {
  "questions": 1000,
  "solve": 0.887,
  "bins": {
   "0": 0.047,
   "1": 0.023,
   "2": 0.112,
   "3": 0.117,
   "4": 0.701
  },
  "median_think": 383.0,
  "p90_think": 1497,
  "cutoff": 0.017375
 },
 "ifeval": {
  "questions": 541,
  "solve": 0.5873382624768947,
  "bins": {
   "0": 0.266173752310536,
   "1": 0.11460258780036968,
   "2": 0.16266173752310537,
   "3": 0.11645101663585952,
   "4": 0.34011090573012936
  },
  "median_think": 300.0,
  "p90_think": 648,
  "cutoff": 0.036044362292051754
 },
 "bfcl_multiple": {
  "questions": 200,
  "solve": 0.760625,
  "bins": {
   "0": 0.13,
   "1": 0.065,
   "2": 0.17,
   "3": 0.075,
   "4": 0.56
  },
  "median_think": 168.0,
  "p90_think": 266,
  "cutoff": 0.0
 },
 "bfcl_irrelevance": {
  "questions": 240,
  "solve": 0.8338541666666667,
  "bins": {
   "0": 0.11666666666666667,
   "1": 0.025,
   "2": 0.07916666666666666,
   "3": 0.05416666666666667,
   "4": 0.725
  },
  "median_think": 128.5,
  "p90_think": 291,
  "cutoff": 0.0
 }
}
