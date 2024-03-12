# AnalyzingCombinedDegradations
Official repository of the paper "Analyzing the Effect of Combined Degradations on Face Recognition"

deg_analysis.py is the Python file in which the trials are made. The code is derived from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py. Moreover, the LFW dataset's binary file and the used ArcFace model's weight also found at the same repository (https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).

trials.zip holds the outputs of all the trials. In addition to the results presented in the paper, cosine distances between the original and degraded images can also be found.
