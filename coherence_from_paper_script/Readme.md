### Coherence Scores for 20NewsGroup  

Since the Github code is slightly different in terms of some parameter settings used in the original paper. So here we
provide two sets of results for `t=50` and `t=200` topics for `prodLDA` on the 20NewsGroup dataset from the code used
in the original paper for reference.

> It is recommended that if you use AVTIM for 20news or RCV1 datasets then use to the settings defined in the paper for
the inference network. Our learning method is closely tied to the architectural settings and operations such as 
`BN` and `dropout` have very specific roles in the optimization. Therefore, for fair comparison it is strongly advised 
that to keep these settings unchanged.
